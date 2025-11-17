import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=15, num_layers=4, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
    def forward(self, x, hidden=None):
        out, (h_n, c_n) = self.lstm(x, hidden)
        h_top = h_n[-1]  # (B, H)
        return out, h_top, (h_n, c_n)

class CNF(nn.Module):
    def __init__(self, data_dim, cond_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        layers = []
        in_dim = data_dim + cond_dim + 1
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers += [nn.Linear(hidden_dim, data_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, s, states):
        # states: [x, cond]
        x = states[:, :self.data_dim]
        cond = states[:, self.data_dim:]
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

# ----------------------------
# Test config
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DA = 4
K  = 10
H  = 15
VMAX = 1150.0  # per paper; used to normalize RMSE to %


ckpt_dir = "/storage/home/hcoda1/3/dwu381/p-yxie77-0/DoFlow/treatment/context_55_prediction_5"
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


def to_t(x): return torch.tensor(x, dtype=torch.float32, device=device)

def rollout_one_option(y_ctx_n, A_ctx, y_loc, y_scale, opt_rows):
    """
    y_ctx_n : (55, 1) normalized
    A_ctx   : (55, DA)
    opt_rows: DataFrame for one patient & one option, times 55..59
              columns: cancer_volume, chemo_application_prev, radio_application_prev
    Returns: rmse (float), preds_n (5,), trues_n (5,)
    """
    # 1) init RNN states from context
    if y_ctx_n.dim() == 1:
        y_ctx_n = y_ctx_n.view(-1, 1)        # (55,1)
    y_ctx_in = y_ctx_n.view(1, -1, 1)        # (1,55,1)
    A_ctx_in = A_ctx.view(1, -1, A_ctx.size(-1))  # (1,55,DA)

    with torch.no_grad():
        _, h_y, hid_y = rnn_y(y_ctx_in)
        _, h_a, hid_a = rnn_a(A_ctx_in)
        K_local = min(K, y_ctx_n.size(0))
        lag = y_ctx_n[-K_local:].view(1, K_local, 1)

    # 2) build 5-step targets and action sequence
    y_true = to_t(opt_rows["cancer_volume"].values).view(5,1)
    y_true_n = (y_true - y_loc) / (y_scale + 1e-8)            # (5,1)

    Capp = to_t(opt_rows["chemo_application_prev"].values).view(5,1)
    Rapp = to_t(opt_rows["radio_application_prev"].values).view(5,1)
    # No dosages in test files; set to 0
    Cdose = torch.zeros_like(Capp)
    Rdose = torch.zeros_like(Rapp)
    A_prev_seq = torch.cat([Capp, Rapp, Cdose, Rdose], dim=1) # (5,DA)

    # 3) autoregressive rollout (B=1)
    preds_n = []
    hy, ha = h_y, h_a
    hy_hid, ha_hid = hid_y, hid_a
    for t in range(5):
        # condition = [h_y_t, h_a_t, lag_K]
        cond = torch.cat([hy, ha, lag.squeeze(-1)], dim=1)  # (1, H+H+K)

        # sample next normalized y
        y_pred_n, _ = cnf_y.sample(cond_states=cond, num_steps=20)
        # advance y-RNN with its own prediction
        y_step_in = y_pred_n.view(1,1,1)                    # (1,1,1)
        _, hy_new, hy_hid = rnn_y(y_step_in, hy_hid)
        hy = hy_new

        # advance action-RNN with provided action for this step
        A_step_in = A_prev_seq[t].view(1,1,DA)              # (1,1,DA)
        _, ha_new, ha_hid = rnn_a(A_step_in, ha_hid)
        ha = ha_new

        # update lag with the *predicted* y (AR decoding)
        lag = torch.cat([lag[:,1:,:], y_pred_n.view(1,1,1)], dim=1)

        preds_n.append(y_pred_n.view(-1))

    preds_n = torch.stack(preds_n, dim=0).view(5,1)  # (5,1)

    # Unnormalize to volume space
    preds_vol = preds_n * (y_scale + 1e-8) + y_loc     # (5,1)
    y_true_vol = y_true_n * (y_scale + 1e-8) + y_loc   # (5,1)

    # Return ONLY the horizon (t+5) values in volume space
    pred_last = float(preds_vol[-1].item())
    true_last = float(y_true_vol[-1].item())
    return pred_last, true_last


# ----------------------------
# Main evaluation loop
# ----------------------------


all_sq_errors = []   # collect ((pred - true)/VMAX)**2 across all patients/options
per_patient_option = defaultdict(list)

patients = ctx_df["patient_id"].unique().tolist()
print(f"#patients in test: {len(patients)}")

for pid in tqdm(patients):
    g_ctx = ctx_df[ctx_df["patient_id"] == pid].sort_values("time")
    y_ctx = to_t(g_ctx["cancer_volume"].values).view(-1,1)
    y_loc = y_ctx.mean()
    y_std = y_ctx.std()
    if torch.isnan(y_std) or y_std < 1e-8:
        y_std = torch.tensor(1e-5, device=device)
    y_ctx_n = (y_ctx - y_loc) / y_std

    Capp = to_t(g_ctx["chemo_application"].values).view(-1,1)
    Rapp = to_t(g_ctx["radio_application"].values).view(-1,1)
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
            y_ctx_n, A_ctx, y_loc, y_std, rows
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
