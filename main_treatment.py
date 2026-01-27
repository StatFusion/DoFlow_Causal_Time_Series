import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchdiffeq import odeint
from torch.nn.utils import clip_grad_value_
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import math
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
from utils import (
    simulate_future_interventional,
    simulate_future_counterfactual,
    quantiles_and_logp,
    metrics,
)
import time
from test_cancer import to_t, rollout_one_option
from collections import defaultdict


print("done importing")


# =========================
# Models & Dataset (unchanged)
# =========================

class MeanScaler(nn.Module):
    """
    Compute mean and standard deviation across time for each feature.
    """
    def __init__(self, keepdim: bool = True, default_scale: float = 1.0):
        super().__init__()
        self.keepdim = keepdim
        self.default_scale = default_scale

    def forward(
        self,
        context: torch.Tensor  # shape: (B, context_length, D)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          - normalized_context: (B, context_length, D)
          - loc: (B, 1, D)  = mean
          - scale: (B, 1, D) = std (or default if too small)
        """
        loc = context.mean(dim=1, keepdim=True)               # (B,1,D)
        var = context.var(dim=1, unbiased=False, keepdim=True)  # (B,1,D)
        scale = var.sqrt().clamp(min=1e-5)                    # (B,1,D)
        scale = torch.where(scale < 1e-5,
                            torch.full_like(scale, self.default_scale),
                            scale)
        norm_ctx = (context - loc) / scale                    # (B,context_length,D)
        return norm_ctx, loc, scale


class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 30, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

    def forward(self, x: torch.Tensor, hidden: tuple = None):
        out, (h_n, c_n) = self.lstm(x, hidden)
        h_top = h_n[-1]  # (B, H)
        return out, h_top, (h_n, c_n)


class CNF(nn.Module):
    def __init__(self, data_dim: int, cond_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.data_dim = data_dim
        self.cond_dim = cond_dim

        layers = []
        in_dim = data_dim + cond_dim + 1 # +1 for flow time s \in [0,1]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        x = states[:, :self.data_dim]
        cond  = states[:, self.data_dim:]
        inp = torch.cat([x, cond, s], dim=1)
        dx  = self.net(inp)
        return dx
    
    @torch.no_grad()
    def sample(self, cond_states, num_steps=20, method='rk4', forward=False, density=False, z0=None):
        # Minimal sampler compatible with training API: Euler fallback for simplicity.
        # Uses fixed-step integration over s in [0,1].
        B = cond_states.size(0)
        D = self.data_dim
        device = cond_states.device
        if z0 is None:
            y = torch.randn(B, D, device=device)
        else:
            y = z0

        # integrate ds (reverse-time for generative sampling)
        steps = num_steps
        dt = 1.0 / steps
        for k in range(steps):
            # reverse-time: s = 1 - t, so use sign = -1
            s = (1.0 - k * dt) if not forward else (k * dt)
            s_t = torch.full((B, 1), s, device=device)
            dy = self.forward(s_t, torch.cat([y, cond_states], dim=1))
            y = y - dy * dt if not forward else y + dy * dt
        return y, None


class CancerTrainDataset(Dataset):
    """
    Windows (context_len, pred_len) over one stochastic node N (cancer_volume),
    with exogenous actions A (chemo/radio) as features.
    """
    def __init__(self, df: pd.DataFrame, context_len=55, pred_len=5):
        self.df = df.sort_values(["patient_id", "time"]).reset_index(drop=True)
        self.context_len = context_len
        self.pred_len    = pred_len
        self.groups = []
        for pid, g in self.df.groupby("patient_id", sort=True):
            g = g.reset_index(drop=True)
            if len(g) >= context_len + pred_len:
                self.groups.append((pid, g))
        self.DA = 4

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        pid, g = self.groups[idx]
        ctx_len, pred_len = self.context_len, self.pred_len
        block = g.iloc[:ctx_len + pred_len].copy()

        # CPU tensors here
        N    = torch.tensor(block["cancer_volume"].values,      dtype=torch.float32)  # (T,)
        Capp = torch.tensor(block["chemo_application"].values,  dtype=torch.float32)
        Rapp = torch.tensor(block["radio_application"].values,  dtype=torch.float32)
        Cdose= torch.tensor(block["chemo_dosage"].values,       dtype=torch.float32)
        Rdose= torch.tensor(block["radio_dosage"].values,       dtype=torch.float32)

        A = torch.stack([Capp, Rapp, Cdose, Rdose], dim=-1)  # (T, DA)

        # splits
        N_ctx, N_fut = N[:ctx_len], N[ctx_len:ctx_len+pred_len]
        A_ctx, A_fut = A[:ctx_len], A[ctx_len:ctx_len+pred_len]

        # scale N using context stats
        loc   = N_ctx.mean()
        scale = N_ctx.std().clamp(min=1e-5)
        N_ctx_n = (N_ctx - loc) / scale
        N_fut_n = (N_fut - loc) / scale

        ctx_in = torch.cat([N_ctx_n.unsqueeze(-1), A_ctx], dim=-1)  # (55, 1+DA)
        fut_in = torch.cat([N_fut_n.unsqueeze(-1), A_fut], dim=-1)  # ( 5, 1+DA)

        return {
            "patient_id": pid,
            "ctx_in": ctx_in,   # CPU tensors
            "fut_in": fut_in,
            "loc": loc.view(1),
            "scale": scale.view(1),
            "N_ctx_n": N_ctx_n,
            "N_fut_n": N_fut_n,
        }



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    DA = 4                # [chemo_app, radio_app, chemo_dose, radio_dose]
    K  = 10               # lags of y
    H  = 15               # hidden size

    df = pd.read_csv("data/cancer_treatment/training_data.csv")\
       .sort_values(["patient_id","time"]).reset_index(drop=True)
    needed = {"patient_id","time","cancer_volume","chemo_application","radio_application","chemo_dosage","radio_dosage"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    train_ds = CancerTrainDataset(df, context_len=55, pred_len=5)
    print(f"#patients with usable windows: {len(train_ds)}")

    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=4,           # OK now—no CUDA work in workers
        pin_memory=pin,
        persistent_workers=True if 4 > 0 else False,
    )


    rnn_y = RNN(input_size=1,   hidden_size=H, num_layers=4).to(device)   # outcome RNN
    rnn_a = RNN(input_size=DA,  hidden_size=H, num_layers=4).to(device)   # actions RNN
    cnf_y  = CNF(data_dim=1, cond_dim=H + H + K, hidden_dim=64, num_layers=3).to(device)
    opt = torch.optim.Adam(list(rnn_y.parameters()) + list(rnn_a.parameters()) + list(cnf_y.parameters()), lr=1e-3)


    N_epochs = 15
    scaler = MeanScaler().to(device)

    for epoch in range(N_epochs):
        rnn_y.train(); rnn_a.train(); cnf_y.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for batch in loop:
            # batch tensors
            ctx_in = batch["ctx_in"].to(device)     # (B,55,1+DA)
            fut_in = batch["fut_in"].to(device)     # (B, 5,1+DA)
            B      = ctx_in.size(0)

            # split
            y_ctx = ctx_in[..., :1]     # (B,55,1)
            A_ctx = ctx_in[..., 1:]     # (B,55,DA)
            y_fut = fut_in[..., :1]     # (B, 5,1) -> normalized targets y_{t+1}
            A_fut = fut_in[..., 1:]     # (B, 5,DA) -> actions driving those futures

            # init optimizer
            opt.zero_grad()

            # 1) run both RNNs over the **context**
            _, h_y, hid_y = rnn_y(y_ctx)     # h_y: (B,H)
            _, h_a, hid_a = rnn_a(A_ctx)     # h_a: (B,H)

            # 2) build lag buffer from last K normalized y’s in context
            if K > 0:
                lag = y_ctx[:, -K:, :]       # (B,K,1)
            else:
                lag = y_ctx[:, :0, :]

            loss = 0.0

            # 3) iterate over the 5 prediction steps
            #    Teacher-forcing on y during training (you can add scheduled sampling if you like)
            hy, ha = h_y, h_a
            hy_hid, ha_hid = hid_y, hid_a

            for t in range(y_fut.size(1)):   # 0..4
                x_true = y_fut[:, t, :]     # (B,1)   # y_{t+1}^{norm}
                A_prev = A_fut[:, t, :]     # (B,DA)  # actions applied to produce x_true

                # CNF condition: [h_y_t, h_a_t, lag_y_K]
                cond = torch.cat([hy, ha, lag.squeeze(-1)], dim=1)  # (B, H+H+K)

                # flow-matching reference
                s   = torch.rand(B, 1, device=device)
                z   = torch.randn(B, 1, device=device)
                x_s = (1 - s) * x_true + s * z

                v_pred = cnf_y.forward(s, torch.cat([x_s, cond], dim=1))
                v_true = z - x_true
                loss  += F.mse_loss(v_pred, v_true)

                # advance both RNNs one step **with teacher forcing** (y_true, A_prev)
                # y-RNN step
                y_step_in = x_true.unsqueeze(1)         # (B,1,1)
                _, hy_new, hy_hid = rnn_y(y_step_in, hy_hid)
                hy = hy_new

                # action-RNN step
                A_step_in = A_prev.unsqueeze(1)         # (B,1,DA)
                _, ha_new, ha_hid = rnn_a(A_step_in, ha_hid)
                ha = ha_new

                # update lag buffer with **true** y (teacher forcing)
                if K > 0:
                    lag = torch.cat([lag[:, 1:, :], x_true.unsqueeze(1)], dim=1)

            loss.backward()
            clip_grad_value_(list(rnn_y.parameters()) + list(rnn_a.parameters()) + list(cnf_y.parameters()), 1.0)
            opt.step()
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1))

        print(f"Epoch {epoch+1:2d} | avg flow-matching loss {total_loss/len(train_loader):.6f}")
        if (epoch+1) % 2 == 0:
            save_dir = "treatment/context_55_prediction_5"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(rnn_y.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}_rnn_y.pth"))
            torch.save(rnn_a.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}_rnn_a.pth"))
            torch.save(cnf_y.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}_cnf_y.pth"))
            print(f"Models saved under {save_dir}")
    


    print("########################################################")
    print("### Done training! Now testing...                      ###")
    print("########################################################")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VMAX = 1150.0  # used to normalize RMSE to % (based on the original paper)


    ckpt_dir = "treatment/context_55_prediction_5"
    ckpt_epoch = 4

    # Saved models
    rnn_y = RNN(input_size=1,  hidden_size=H, num_layers=4).to(device)
    rnn_a = RNN(input_size=DA, hidden_size=H, num_layers=4).to(device)
    cnf_y  = CNF(data_dim=1, cond_dim=H + H + K, hidden_dim=64, num_layers=3).to(device)

    rnn_y.load_state_dict(torch.load(os.path.join(ckpt_dir, f"epoch_{ckpt_epoch}_rnn_y.pth"), map_location=device))
    rnn_a.load_state_dict(torch.load(os.path.join(ckpt_dir, f"epoch_{ckpt_epoch}_rnn_a.pth"), map_location=device))
    cnf_y.load_state_dict(torch.load(os.path.join(ckpt_dir, f"epoch_{ckpt_epoch}_cnf_y.pth"), map_location=device))

    rnn_y.eval(); rnn_a.eval(); cnf_y.eval()


    ctx_path  = "data/cancer_treatment/test_sequence_context_t0_54.csv"
    final5_path = "data/cancer_treatment/test_sequence_final5.csv"

    ctx_df   = pd.read_csv(ctx_path).sort_values(["patient_id","time"]).reset_index(drop=True)
    final_df = pd.read_csv(final5_path).sort_values(["patient_id","option_id","time"]).reset_index(drop=True)

    # Sanity: expect time 0..54 in ctx; time 55..59 in final
    assert ctx_df["time"].min() == 0 and ctx_df["time"].max() == 54, "Unexpected ctx time range"
    assert final_df["time"].min() == 55 and final_df["time"].max() == 59, "Unexpected final5 time range"

    all_sq_errors = []   # collect ((pred - true)/VMAX)**2 across all patients/options
    per_patient_option = defaultdict(list)

    patients = ctx_df["patient_id"].unique().tolist()
    print(f"#patients in test: {len(patients)}")

    for pid in tqdm(patients):
        g_ctx = ctx_df[ctx_df["patient_id"] == pid].sort_values("time")
        y_ctx = to_t(g_ctx["cancer_volume"].values, device=device).view(-1,1)
        y_loc = y_ctx.mean()
        y_std = y_ctx.std()
        if torch.isnan(y_std) or y_std < 1e-8:
            y_std = torch.tensor(1e-5, device=device)
        y_ctx_n = (y_ctx - y_loc) / y_std

        Capp = to_t(g_ctx["chemo_application"].values, device=device).view(-1,1)
        Rapp = to_t(g_ctx["radio_application"].values, device=device).view(-1,1)
        Cdose = torch.zeros_like(Capp)
        Rdose = torch.zeros_like(Rapp)
        A_ctx = torch.cat([Capp, Rapp, Cdose, Rdose], dim=1)

        g_fin = final_df[final_df["patient_id"] == pid]
        if g_fin.empty:
            continue
        option_ids = sorted(g_fin["option_id"].unique().tolist())

        for oid in option_ids:
            rows = g_fin[g_fin["option_id"] == oid].sort_values("time")
            if len(rows) != 5:
                continue

            # Get final-step predicted & true volumes (not normalized)
            pred_last_vol, true_last_vol = rollout_one_option(
                y_ctx_n, A_ctx, y_loc, y_std, rows,
                rnn_y, rnn_a, cnf_y, device, K, DA
            )

            # squared error, normalized by VMAX
            se_norm = ((pred_last_vol - true_last_vol) / VMAX) ** 2
            all_sq_errors.append(se_norm)
            per_patient_option[pid].append((oid, math.sqrt(se_norm) * 100.0))

    # ----------------------------
    # Reporting
    # ----------------------------
    print("\n=== Per-patient per-option NRMSE at horizon (%, optional preview) ===")
    for pid in sorted(per_patient_option.keys())[:10]:  # preview first 10 patients
        entries = sorted(per_patient_option[pid], key=lambda x: x[0])
        s = ", ".join([f"opt{oid}: {nrmse:.3f}%" for oid, nrmse in entries])
        print(f"patient {pid}: {s}")

    if all_sq_errors:
        rmse_norm_pct = 100.0 * math.sqrt(float(np.mean(all_sq_errors)))
    else:
        rmse_norm_pct = float("nan")

    print(f"\nOverall Normalized RMSE at t+5 (as % of Vmax={VMAX}): {rmse_norm_pct:.4f}%")
