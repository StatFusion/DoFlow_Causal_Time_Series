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
    """
    Autoregressive single-node LSTM.
    Takes [x_i, time_features] for one node as input.
    """
    def __init__(self,
                 time_feat_dim: int,
                 hidden_size: int = 15,
                 num_layers: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        self.time_feat_dim = time_feat_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size = 1 + time_feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

    def forward(self, x: torch.Tensor, hidden: tuple = None):
        out, (h_n, c_n) = self.lstm(x, hidden)
        h_top = h_n[-1]  # (batch_size, hidden_size)
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
    
    def sample(self,
               cond_states: torch.Tensor,   # (B, cond_dim)
               num_steps: int = 50,
               method: str = 'rk4',
               forward: bool = False,
               density: bool = False,
               z0: Optional[torch.Tensor] = None):
        B, _ = cond_states.shape
        device = cond_states.device

        D = self.data_dim
        if z0 is None:
            z0 = torch.randn(B, D, device=device)
            y0 = z0
        else:
            y0 = z0

        logp0  = -0.5 * (y0**2).sum(dim=1, keepdim=True) - (D/2) * math.log(2 * math.pi)
        M = 1
        sigma = 5e-3
        eps = torch.randn(M, B, D, device=device) if density else None

        def ode_func(t, states):
            if density:
                y, logp = states
            else:
                y = states
            batch = y.shape[0]
            if forward:
                s = (t).reshape(1,1).expand(batch,1)
                sign = 1
            else:
                s = (1 - t).reshape(1,1).expand(batch,1)
                sign = -1
            
            inp = torch.cat([y, cond_states], dim=1)
            dx  = self.forward(s, inp)
            if density:
                y_eps = y.unsqueeze(0).expand(M, B, D) + sigma * eps
                cond_eps = cond_states.unsqueeze(0).expand(M, B, self.cond_dim)
                y_flat    = y_eps.reshape(M*B, D)
                cond_flat = cond_eps.reshape(M*B, self.cond_dim)
                s_flat    = s.unsqueeze(0).expand(M, batch, 1).reshape(M*B, 1)

                states_flat = torch.cat([y_flat, cond_flat], dim=1)
                dx_flat     = self.forward(s_flat, states_flat)
                dx_eps      = dx_flat.reshape(M, B, D)
                diff = (dx_eps - dx.unsqueeze(0)) / sigma
                trace = (eps * diff).sum(dim=2).mean(dim=0, keepdim=True)
                dydt  = sign * dx
                dlogp = -sign * trace.T
                return dydt, dlogp
            else:
                return sign * dx

        t_grid = torch.linspace(0.0, 1.0, num_steps, device=device)
        if density:
            yT, logpT = odeint(ode_func, (y0, logp0), t_grid, method=method)
            return yT[-1], logpT[-1]
        else:
            yT = odeint(ode_func, y0, t_grid, method=method)
            return yT[-1], None


class SimulatedDataset(Dataset):
    """
    Clean simulated time series with context/prediction splits.
    """
    def __init__(self, df, context_length: int, prediction_length: int, device: torch.device, stride: int = 1):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.device = device
        self.stride = max(1, stride)
        if hasattr(df, 'values'):
            data = torch.tensor(df.values, dtype=torch.float32, device=device)
        else:
            data = df.to(device=device, dtype=torch.float32)
        self.raw = data  # (N, D)

        N = self.raw.size(0)
        total_len = context_length + prediction_length
        self.indices = list(range(0, N - total_len + 1, self.stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        mid = start + self.context_length
        end = mid + self.prediction_length
        x = self.raw[start:mid]
        y = self.raw[mid:end]
        return torch.cat([x, y], dim=0)


def topo_sort(parents: Dict[int, List[int]]) -> List[int]:
    visited = set()
    order = []
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for p in parents[node]:
            dfs(p)
        order.append(node)
    for node in parents:
        dfs(node)
    return order


# =========================
# Inference/Test helpers (new; factorized)
# =========================

def _prepare_test_batch(test_ds, idxs, context_length, device):
    X_batch = torch.stack([test_ds[i] for i in idxs], dim=0).to(device)
    ctx = X_batch[:, :context_length, :]
    fut = X_batch[:, context_length:, :]
    return ctx, fut

def _scale_context(ctx, D, scaler: MeanScaler, K: int):
    """
    Always scale; returns:
      ctx_in: (B, ctx, D),
      norm_ctx: (B, ctx, D),
      loc: (B,1,D), scale: (B,1,D),
      buf0: (B, K, D)
    """
    ctx_raw = ctx[..., :D]
    # ctx_time = ctx[..., D:]
    norm_ctx, loc, scale = scaler(ctx_raw)
    # ctx_in = torch.cat([norm_ctx, ctx_time], dim=-1)
    ctx_in = norm_ctx
    buf0 = norm_ctx[:, -K:, :] if K > 0 else norm_ctx[:, :0, :]
    return ctx_in, norm_ctx, loc, scale, buf0

def _init_node_rnns(rnn_list, ctx_in, D, num_ens, K, buf0):
    """
    Run initial pass over context to get per-node hidden states.
    Returns:
      hidden: list of (h_n, c_n) per node (broadcast to ensembles)
      h_list: dict node->top hidden per chain
      condition_x_last: (B*num_ens, K, D)
      num_ens: number of predictions to generate for each test time series.
      B_e = B * num_ens
    """
    B, context_length, _ = ctx_in.shape
    # TF = ctx_in.shape[-1] - D

    hidden = [None] * D
    h_list = {}

    for i, rn in enumerate(rnn_list):
        node_seq = ctx_in[:, :, i:(i+1)]
        # time_feats = ctx_in[:, :, D:]
        # inp = torch.cat([node_seq, time_feats], dim=-1)
        inp = node_seq
        _, h_top, hid = rn(inp, None)
        h_n, c_n = hid
        h_n = h_n.repeat_interleave(num_ens, dim=1)
        c_n = c_n.repeat_interleave(num_ens, dim=1)
        hidden[i] = (h_n, c_n)
        h_list[i] = h_top.repeat_interleave(num_ens, dim=0)

    buf0_e = buf0.unsqueeze(1).expand(B, num_ens, buf0.size(1), D)
    B_e = B * num_ens
    condition_x_last = buf0_e.reshape(B_e, buf0.size(1), D).clone()
    return hidden, h_list, condition_x_last, B_e


def _choose_intervention_set(Type: str):
    return [0,1,2] if (Type == "layer_nonlinear") else [0]

def _rollout_observational(
    fut, parents, order, rnn_list, cnfs,
    hidden, h_list, condition_x_last, B, B_e, D, num_ens, prediction_length
):
    """
    Pure sampling rollout (observational/forecasting).
    Returns: samples_arr_torch (T,B,E,D) as a torch tensor on the same device.
    """
    device = fut.device
    all_ens = []
    with torch.no_grad():
        for t in range(prediction_length):
            x_pred_t = {}
            h_next = {}
            # prediction proceeds in topological order (parents first; children last)
            for i in order:
                parts = [h_list[i]]
                if condition_x_last.size(1) > 0:
                    parts.append(condition_x_last[:, :, i])
                if parents[i]:
                    pv = torch.cat([condition_x_last[:, -1, p].unsqueeze(1) for p in parents[i]], dim=1)
                    parts.append(pv)
                    x_parents_hidden_list = [h_list[j] for j in parents[i]]
                    parts.extend(x_parents_hidden_list)

                cond_i = torch.cat(parts, dim=1)
                x_i, _ = cnfs[i].sample(cond_i, density=False)   # stays on device
                x_pred_t[i] = x_i

                node_seq_e = x_i.view(B_e, 1, 1)
                rn = rnn_list[i]
                _, h_top, hid = rn(node_seq_e, hidden[i])
                h_next[i] = h_top
                hidden[i] = hid

            h_list = h_next
            x_pred = torch.cat([x_pred_t[i] for i in range(D)], dim=1)  # (B*E, D)
            all_ens.append(x_pred)
            condition_x_last = torch.cat([condition_x_last[:, 1:, :], x_pred.unsqueeze(1)], dim=1)

    samples = torch.stack(all_ens, dim=0).cpu().numpy()                # (T, B*E, D)
    samples_arr = samples.reshape(prediction_length, B, num_ens, D)  # (T,B,E,D)
    return samples_arr


def _counterfactual_forward_pass(
    mode, prediction_length, true_future, D, B, num_ens,
    rnn_list, cnfs, parents, order,
    hidden, h_list, condition_x_last, loc, scale
):
    """
    Returns dictionaries needed for CF latent reuse (z_all) and advanced hidden/buffer states.
    """
    if mode != "counterfactual":
        return None, None, None, None

    with torch.no_grad():
        hidden_tf = [h for h in hidden]
        h_list_tf = {i: h_list[i].clone() for i in h_list}
        cond_buffer  = condition_x_last.clone()
        z_all = {i: [] for i in range(D)}
        x_true_t = {}

        for t in range(prediction_length):

            h_prev = {k: v for k, v in h_list_tf.items()} ##

            for i in order:
                true_vals = (true_future[:, t, i] - loc[:,0,i]) / scale[:,0,i]


                # parts = [h_list_tf[i]]
                parts = [h_prev[i]]

                if cond_buffer.size(1) > 0:
                    parts.append(cond_buffer[:, :, i])
                if parents[i]:
                    pv = torch.cat([cond_buffer[:, -1, p].unsqueeze(1) for p in parents[i]], dim=1)
                    parts.append(pv)
                    # x_parents_hidden_list = [h_list_tf[j] for j in parents[i]]
                    x_parents_hidden_list = [h_prev[j] for j in parents[i]]

                    parts.extend(x_parents_hidden_list)

                cond_inv = torch.cat(parts, dim=1)
                x_true = true_vals.unsqueeze(-1).unsqueeze(-1).expand(B, num_ens, 1).reshape(B * num_ens, 1)
                x_true_t[i] = x_true
                z_i, _ = cnfs[i].sample(cond_inv, density=False, forward=True, z0=x_true)
                z_all[i].append(z_i)

                node_seq_e = x_true.view(B * num_ens, 1, 1)
                out, h_top, hid = rnn_list[i](node_seq_e, hidden_tf[i])
                hidden_tf[i], h_list_tf[i] = hid, h_top

            new_buf = torch.cat([x_true_t[i] for i in range(D)], dim=1)
            cond_buffer = torch.cat([cond_buffer[:, 1:, :], new_buf.unsqueeze(1)], dim=1)

    return hidden_tf, h_list_tf, cond_buffer, z_all

def _rollout_interv_cf(
    fut, mode, inter_set, parents, order, rnn_list, cnfs,
    hidden, h_list, condition_x_last, B, B_e, D, num_ens,
    prediction_length, z_all, loc, scale
):
    """
    Interventional / Counterfactual rollout prediction.

    fut: the simulated true future interventional/counterfactual values.
         Used to intervene the root nodes with the simulated true values.
    B: batch size
    num_ens: number of predictions to generate for each test time series.
    B_e = B * num_ens
    hidden, h_list: the LSTM hidden states that summarize the Context histories before the forecasting window
    """
    all_ens = []

    with torch.no_grad():
        for t in range(prediction_length):

            x_pred_t, x_pred_t_MAP, logp_t, h_next = {}, {}, {}, {}

            # prediction proceeds in topological order (parents first; children last)
            for i in order:
                if i in inter_set:
                    ### Intervening the root nodes with the simulated true values
                    true_vals_raw = fut[:, t, i]
                    true_vals_norm = (true_vals_raw - loc[:,0,i]) / scale[:,0,i]
                    
                    true_col  = true_vals_norm.unsqueeze(-1)
                    true_brd  = true_col.unsqueeze(1).expand(B, num_ens, 1)
                    flat_true = true_brd.reshape(B * num_ens, 1)

                    x_pred_t[i] = flat_true
                    logp_t[i]   = torch.zeros_like(flat_true)
                    x_pred_t_MAP[i] = flat_true

                    node_seq_e = true_brd.reshape(B * num_ens, 1, 1)
                    _, h_top, hid = rnn_list[i](node_seq_e, hidden[i])
                    h_next[i]  = h_top
                    hidden[i]  = hid
                    continue

                #### Predict the non-intervened (non-root) nodes
                # condition_x_last: the last K values of the nodes' predicted past, i.e., \hat{X}_{t-K:t-1}
                # h_list: the LSTM hidden states that summarize the histories up to t-1
                parts = [h_list[i]]
                if condition_x_last.size(1) > 0:
                    parts.append(condition_x_last[:, :, i]) # condition includes the node's past K values
                if parents[i]:
                    # condition includes its parents' last-step values
                    pv = torch.cat([condition_x_last[:, -1, p].unsqueeze(1) for p in parents[i]], dim=1)
                    parts.append(pv)
                    # condition includes its parents' last-step hidden states
                    x_parents_hidden_list = [h_list[j] for j in parents[i]]
                    parts.extend(x_parents_hidden_list)

                cond_i = torch.cat(parts, dim=1)
                if mode == "interventional":
                    # By default, density = False; sampling time would be much longer if set density = True
                    x_i, logp_i = cnfs[i].sample(cond_i, density=False)
                else:
                    z0 = z_all[i][t]
                    x_i, logp_i = cnfs[i].sample(cond_i, density=False, forward=False, z0=z0)

                x_pred_t[i] = x_i
                logp_t[i] = logp_i
                node_seq_e = x_i.view(B_e, 1, 1)
                _, h_top, hid = rnn_list[i](node_seq_e, hidden[i])
                h_next[i] = h_top
                hidden[i] = hid

            h_list = h_next
            x_pred = torch.cat([x_pred_t[i] for i in range(D)], dim=1)
            all_ens.append(x_pred)


            condition_x_last = torch.cat([condition_x_last[:, 1:, :], x_pred.unsqueeze(1)], dim=1)

    samples = torch.stack(all_ens, dim=0).cpu().numpy()
    samples_arr = samples.reshape(prediction_length, B, num_ens, D)
    return samples_arr



# =========================
# Testing
# =========================

def test_observational(
    test_ds, rnn_list, cnfs, epoch, parents,
    context_length, prediction_length, device,
    K, output_dir, num_ens=1, scaler=None, use_parents=True, stride=1, Type="chain", mode="forecasting"
):
    """
    Observational forecasting rollout with fan charts.
    """
    # eval mode
    for rn in rnn_list: rn.eval()
    for cnf in cnfs: cnf.eval()
    order = topo_sort(parents)

    # sample windows
    random_idxs = torch.randint(0, 1000, (1000,)).tolist()
    idxs = random_idxs

    D = len(parents)
    B = len(idxs)

    # batch + scale
    ctx, fut = _prepare_test_batch(test_ds, idxs, context_length, device)
    ctx_in, norm_ctx, loc, scale, buf0 = _scale_context(ctx, D, scaler, K)

    # init
    hidden, h_list, condition_x_last, B_e = _init_node_rnns(rnn_list, ctx_in, D, num_ens, K, buf0)

    # rollout
    t0 = time.time()
    samples_arr = _rollout_observational(fut, parents, order, rnn_list, cnfs,
                                         hidden, h_list, condition_x_last,
                                         B, B_e, D, num_ens, prediction_length)
    print(f"Time taken: {time.time() - t0} seconds")

    _, _, _, q5, q10, q25, q50, q75, q90, q95 = quantiles_and_logp(
        None, samples_arr, device, loc, scale
    )
    q_dict = {5:q5.cpu().numpy(), 10:q10.cpu().numpy(), 25:q25.cpu().numpy(), 50:q50.cpu().numpy(), 75:q75.cpu().numpy(), 90:q90.cpu().numpy(), 95:q95.cpu().numpy()}
    metrics(fut, samples_arr, device, loc, scale, output_dir, mode)
    
    # _plot_fan_charts_observational(
    #     samples_arr, q_dict, ctx, norm_ctx, loc, scale,
    #     context_length, prediction_length, fut, D, output_dir, idxs, num_ens, mode=mode
    # )


def test_interv_cf(
    test_ds, rnn_list, cnfs, epoch, parents,
    context_length, prediction_length, device,
    K, output_dir, num_ens=1, scaler=None, use_parents=True, stride=1, Type='chain', mode="counterfactual"
):
    """
    num_ens: number of predictions to generate for each test series. (The flow can generate multiple samples for each series.)
    """
    for rn in rnn_list: rn.eval()
    for cnf in cnfs: cnf.eval()
    order = topo_sort(parents)

    inter_set = _choose_intervention_set(Type)

    random_idxs = torch.randint(0, 1000, (1000,)).tolist()
    idxs = random_idxs

    D = len(parents)
    B = len(idxs)
    ctx, fut_obs = _prepare_test_batch(test_ds, idxs, context_length, device)
    ctx_in, norm_ctx, loc, scale, buf0 = _scale_context(ctx, D, scaler, K)

    true_future = fut_obs.clone()

    #### Simulate the true future interventional/counterfactual values
    if mode == "interventional":
        fut = simulate_future_interventional(Type, B, D, parents, prediction_length, device, ctx, true_future).to(device) # unscaled future
    elif mode == "counterfactual":
        fut = simulate_future_counterfactual(Type, ctx, parents, prediction_length, device, true_future, inter_set).to(device)
    else:
        raise ValueError("mode must be 'interventional' or 'counterfactual'")

    hidden, h_list, condition_x_last, B_e = _init_node_rnns(rnn_list, ctx_in, D, num_ens, K, buf0)
    if mode == "counterfactual":
        _, _, _, z_all = _counterfactual_forward_pass(
            mode, prediction_length, true_future, D, B, num_ens,
            rnn_list, cnfs, parents, order, hidden, h_list, condition_x_last, loc, scale
        )
    else:
        z_all = None

    #### Run rollout prediction over the forecasting window
    # samples_arr: predicted future interventional/counterfactual values
    samples_arr = _rollout_interv_cf(
        fut, mode, inter_set, parents, order, rnn_list, cnfs,
        hidden, h_list, condition_x_last, B, B_e, D, num_ens,
        prediction_length, z_all,
        loc, scale
    )
    
    _, _, _, q5, q10, q25, q50, q75, q90, q95 = quantiles_and_logp(
        None, samples_arr, device, loc, scale
    )
    q_dict = {5:q5.cpu().numpy(), 10:q10.cpu().numpy(), 25:q25.cpu().numpy(), 50:q50.cpu().numpy(), 75:q75.cpu().numpy(), 90:q90.cpu().numpy(), 95:q95.cpu().numpy()}
    metrics(fut, samples_arr, device, loc, scale, output_dir, mode)



    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"epoch_{epoch}_samples_arr.npy"), samples_arr)
    np.save(os.path.join(output_dir, f"epoch_{epoch}_idxs.npy"), np.array(idxs, dtype=np.int64))
    np.save(os.path.join(output_dir, f"epoch_{epoch}_norm_ctx.npy"), norm_ctx.detach().cpu().numpy())   # (B,ctx,D)
    np.save(os.path.join(output_dir, f"epoch_{epoch}_loc.npy"),      loc.detach().cpu().numpy())        # (B,1,D)
    np.save(os.path.join(output_dir, f"epoch_{epoch}_scale.npy"),    scale.detach().cpu().numpy())      # (B,1,D)
    np.save(os.path.join(output_dir, f"epoch_{epoch}_true_future.npy"), true_future.detach().cpu().numpy())  # (B,pred,D)
    np.save(os.path.join(output_dir, f"epoch_{epoch}_fut.npy"),         fut.detach().cpu().numpy())          # (B,pred,D)
    np.save(os.path.join(output_dir, f"epoch_{epoch}_inter_set.npy"), np.array(inter_set, dtype=np.int64))
    meta = {
        "mode": mode,
        "Type": Type,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "K": K,
        "num_ens": num_ens,
        "D": D,
        "B": B,
    }
    np.save(os.path.join(output_dir, f"epoch_{epoch}_meta.npy"), meta, allow_pickle=True)
    np.save(os.path.join(output_dir, f"epoch_{epoch}_q_dict.npy"), q_dict, allow_pickle=True)




    _plot_interv_cf(samples_arr, q_dict, norm_ctx, loc, scale,
             context_length, prediction_length, fut, true_future,
             D, output_dir, idxs, inter_set, mode, epoch=epoch)

def main_test(
    Type: str,
    mode: str,                   # "forecasting" | "interventional" | "counterfactual"
    parents: dict,
    data_path: str,                          # if None -> uses default template below
    save_dir: str,                           # if None -> auto from Type/lengths/stride/K/use_parents
    output_dir: str,                         # if None -> auto from Type/context/pred/K
    context_length: int = 90,
    prediction_length: int = 30,
    stride: int = 1,
    K: int = 10,
    hidden_size: int = 15,
    epoch: int = 50,
    num_ens: int = 30,
    use_parents: bool = True,
    device: torch.device = None,
):
    """
    Runs the full testing pipeline:
      - loads dataset split
      - reconstructs models (RNNs + CNFs) and loads checkpoints
      - dispatches to forecasting or interventional/counterfactual testing
      - writes plots/metrics to output_dir
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_path is None:
        data_path = f"simulation/simulated_data/simulated_timeseries_{Type}.csv"

    if save_dir is None:
        save_dir = (
            f"simulation/"
            f"trained_model_simulation/{Type}_hidden_par_context_{context_length}_prediction_{prediction_length}"
        )

    if output_dir is None:
        output_dir = (
            f"simulation/results/"
            f"{Type}_{context_length}_{prediction_length}_last_{K}"
        )
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    split_idx = int(len(df) * 0.8)
    test_df  = df.iloc[split_idx:].reset_index(drop=True)

    test_ds  = SimulatedDataset(test_df, context_length, prediction_length, device, stride=stride)

    D  = len(parents)

    rnn_list = torch.nn.ModuleList(
        [RNN(time_feat_dim=0, num_layers=3, hidden_size=hidden_size).to(device) for _ in range(D)]
    )
    for i, rn in enumerate(rnn_list):
        pattern = os.path.join(save_dir, f"epoch_{epoch}_rnn_node_{i}.pth")
        rn.load_state_dict(torch.load(pattern, map_location=device))
        rn.eval()

    # CNFs
    cnfs = []
    for i in range(D):
        if use_parents:
            # hidden of this node + (hidden + value) per parent + K lag
            cond_dim = rnn_list[i].hidden_size + len(parents[i]) * (hidden_size + 1) + K
        else:
            cond_dim = rnn_list[i].hidden_size + K
        cnf = CNF(data_dim=1, cond_dim=cond_dim, hidden_dim=64, num_layers=3).to(device)
        pattern = os.path.join(save_dir, f"epoch_{epoch}_cnf_node_{i}.pth")
        cnf.load_state_dict(torch.load(pattern, map_location=device))
        cnf.eval()
        cnfs.append(cnf)

    scaler = MeanScaler().to(device)

    # -------------------
    # run chosen test
    # -------------------
    if mode == "forecasting":
        test_observational(
            test_ds=test_ds,
            rnn_list=rnn_list,
            cnfs=cnfs,
            epoch=epoch,
            parents=parents,
            context_length=context_length,
            prediction_length=prediction_length,
            device=device,
            K=K,
            output_dir=output_dir,
            num_ens=num_ens,
            scaler=scaler,
            use_parents=use_parents,
            stride=stride,
            Type=Type,
            mode=mode,
        )
    elif mode in ("interventional", "counterfactual"):
        test_interv_cf(
            test_ds=test_ds,
            rnn_list=rnn_list,
            cnfs=cnfs,
            epoch=epoch,
            parents=parents,
            context_length=context_length,
            prediction_length=prediction_length,
            device=device,
            K=K,
            output_dir=output_dir,
            num_ens=num_ens,
            scaler=scaler,
            use_parents=use_parents,
            stride=stride,
            Type=Type,
            mode=mode,
        )
    else:
        raise ValueError("mode must be 'forecasting', 'interventional', or 'counterfactual'")

    del rnn_list, cnfs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _plot_fan_charts_observational(
    samples_arr, q_inv, ctx, ctx_raw, loc, scale,
    context_length, prediction_length, fut, D, output_dir, idxs, num_ens, mode="forecasting"
):
    B = samples_arr.shape[1]
    random_chain = np.random.randint(num_ens, size=B)
    for b in range(min(10, B)):
        rows = 2 if D <= 8 else (D + 4) // 5
        cols = 4 if D <= 8 else 5
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        axes = axes.flatten()

        ctx_b = (ctx_raw[b] * scale[b] + loc[b]).cpu().numpy()
        true_b = fut[b, :, :D].cpu().numpy()

        t_ctx = np.arange(context_length - context_length//4, context_length)
        t_pred = np.arange(context_length, context_length+prediction_length)

        for i in range(D):
            ax = axes[i]
            ax.plot(t_ctx, ctx_b[-(context_length//4):, i], label="context")
            ax.plot(t_pred, true_b[:, i], label="true")
            ax.fill_between(t_pred, q_inv[5][:, b, i],  q_inv[95][:, b, i], alpha=0.15, label="90% CI")
            ax.fill_between(t_pred, q_inv[25][:, b, i], q_inv[75][:, b, i], alpha=0.3,  label="50% CI")
            ax.legend(fontsize="small")
        for j in range(D, len(axes)):
            axes[j].axis("off")

        os.makedirs(output_dir, exist_ok=True)
        # np.save(os.path.join(output_dir, f"samples_arr_{mode}.npy"), samples_arr)
        fig.savefig(os.path.join(output_dir, f"observational_plot_idx_{idxs[b]}.png"))
        plt.close(fig)


def _plot_interv_cf(
    samples_arr, q_dict, ctx_raw, loc, scale,
    context_length, prediction_length, fut, true_future,
    D, output_dir, idxs, inter_set, mode, epoch=None
):
    B = samples_arr.shape[1]
    mean_samples = samples_arr.mean(axis=2)
    loc_np   = loc.cpu().numpy().squeeze(1)
    scale_np = scale.cpu().numpy().squeeze(1)

    for b in range(min(5, len(idxs))):
        rows = 2 if D <= 8 else (D + 4) // 5
        cols = 4 if D <= 8 else 5
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        fig.suptitle('Interventional Forecasting' if mode == "interventional" else 'Counterfactual Forecasting',
                     fontsize=23, weight='bold')
        axes = axes.flatten()

        ctx_b = (ctx_raw[b] * scale[b] + loc[b]).cpu().numpy()
        true_b = fut[b, :, :D].cpu().numpy()     # simulated target (int./cf baseline)

        true_future_b = true_future[b, :, :D].cpu().numpy()

        t_ctx = np.arange(context_length - context_length//4, context_length)
        t_pred = np.arange(context_length, context_length+prediction_length)

        for i in range(D):
            ax = axes[i]
            ax.plot(t_ctx, ctx_b[-(context_length//4):, i], label="Context")
            ax.plot(t_pred, true_b[:, i], label=("Conducted Int." if i in inter_set else ("Int. Future" if mode=="interventional" else "CF. Future")), color="orange")
            ax.plot(t_pred, true_future_b[:, i], label="Obs. Future", linestyle=':', linewidth=0.9, color='gray', alpha=0.8)

            if mode == "counterfactual" and i not in inter_set:
                sample_chain = samples_arr[:, b, 0, i] * scale_np[b, i] + loc_np[b, i]
                ax.plot(t_pred, sample_chain, label="Est. CF.", color="mediumseagreen")

            if i >= len(inter_set) and mode == "interventional":
                ax.fill_between(t_pred, q_dict[5][:, b, i],  q_dict[95][:, b, i], alpha=0.15, label="Est. Int. 90% CI")
                ax.fill_between(t_pred, q_dict[25][:, b, i], q_dict[75][:, b, i], alpha=0.3,  label="Est. Int. 50% CI")

            ax.axvline(context_length, ls="--", color="k", lw=1, alpha=0.7, label=("Int. start" if mode=="interventional" and i==0 else ("CF. start" if mode!="interventional" and i==0 else None)))
            ax.set_title(rf"$X_{{{i+1},t}} (Intervened)$" if i < len(inter_set) else rf"$X_{{{i+1},t}}$", fontsize=22)

            if i == 0 or i == D//2:
                ax.legend(fontsize=12.5, loc="lower left", framealpha=0.7)
            else:
                ax.legend().set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=10.5)

        for j in range(D, len(axes)):
            axes[j].axis("off")

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"epoch_{epoch}_interv_plot_idx_{idxs[b]}.png" if mode == "interventional" else f"epoch_{epoch}_cf_plot_idx_{idxs[b]}.png")
        pdf_save_path = save_path.replace(".png", ".pdf")
        fig.savefig(pdf_save_path, bbox_inches="tight")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)


# =========================
# Training
# =========================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    Type = "tree_linear"
    # Type = "layer_nonlinear"
    # Type = "diamond_linear"
    # Type = "chain_linear"
    # Type = "chain_nonlinear"

    path = f'data/simulated_data/simulated_timeseries_{Type}.csv'
    df = pd.read_csv(path)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df  = df.iloc[split_idx:].reset_index(drop=True)

    context_length    = 90
    prediction_length = 30
    stride            = 1

    train_ds = SimulatedDataset(train_df, context_length, prediction_length, device, stride=stride)
    test_ds  = SimulatedDataset(test_df,  context_length, prediction_length, device, stride=stride)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)
    print(f"Training series: {len(train_ds)}")
    print(f"Testing  series: {len(test_ds)}")

    if Type == "chain_linear" or Type == "chain_nonlinear":
        parents = {i: ([] if i == 0 else ([0] if i == 1 else [0, i-1])) for i in range(50)}
    elif Type == "tree_linear" or Type.startswith("tree_linear"):
        parents = {0: [], 1:[0], 2:[0], 3:[1], 4:[1], 5:[2], 6:[2], 7:[6]}
    elif Type == "diamond_linear":
        parents = {0: [], 1:[0], 2:[0], 3:[1], 4:[2], 5:[1], 6:[2], 7:[3,5], 8:[4,6], 9:[7,8]}
    elif Type == "layer_nonlinear":
        parents = {0: [], 1:[], 2:[], 3:[0,1,2], 4:[0,1,2], 5:[0,1,2], 6:[0,1,2], 7:[3,4,5,6], 8:[3,4,5,6], 9:[3,4,5,6]}

    use_parents = True
    D = len(parents)
    K = 10 # number of previous time-steps values to use as conditioning
    hidden_size = 15

    rnn_list = nn.ModuleList([RNN(time_feat_dim=0, num_layers=3, hidden_size=hidden_size).to(device) for _ in range(D)])
    
    cnfs = []
    cnf_layers = 3
    for i in range(D):
        if use_parents:
            cond_dim = rnn_list[i].hidden_size + len(parents[i])*hidden_size + len(parents[i]) + K
        else:
            cond_dim = rnn_list[i].hidden_size + K
        cnf = CNF(data_dim=1, cond_dim=cond_dim, hidden_dim=64, num_layers=cnf_layers).to(device)
        cnfs.append(cnf)
    
    scaler = MeanScaler().to(device)

    opt = torch.optim.Adam(
        [p for rn in rnn_list for p in rn.parameters()] + [p for cnf in cnfs for p in cnf.parameters()],
        lr= 1e-3
    )

    #########################################################
    # Training
    #########################################################
    save_dir = f"simulation/trained_model_simulation/{Type}_hidden_par_context_{context_length}_prediction_{prediction_length}"
    os.makedirs(save_dir, exist_ok=True)
    N_epochs = 10

    # if os.path.exists(os.path.join(save_dir, f"epoch_{N_epochs}_rnn_node_0.pth")):
    #     print(f"Models already exist. Loading models from {save_dir}")
    #     for i, rn in enumerate(rnn_list):
    #         rn.load_state_dict(torch.load(os.path.join(save_dir, f"epoch_{N_epochs}_rnn_node_{i}.pth"), map_location=device))
    #         rn.eval()
    #     for i, cnf in enumerate(cnfs):
    #         cnf.load_state_dict(torch.load(os.path.join(save_dir, f"epoch_{N_epochs}_cnf_node_{i}.pth"), map_location=device))
    #         cnf.eval()
    #     # print("Running test on observational forecasting...")
    #     output_dir = f"simulation/results/{Type}_hidden_par_context_{context_length}_prediction_{prediction_length}"
    #     # main_test(Type=Type, mode="forecasting", parents=parents, data_path=path, save_dir=save_dir, output_dir=output_dir, context_length=context_length,
    #     #     prediction_length=prediction_length, stride=stride, K=K, hidden_size=hidden_size, epoch=N_epochs, num_ens=30, use_parents=use_parents, device=device
    #     # )

    #     # print("Running test on interventional...")
    #     # main_test(Type=Type, mode="interventional", parents=parents, data_path=path, save_dir=save_dir, output_dir=output_dir, context_length=context_length,
    #     #     prediction_length=prediction_length, stride=stride, K=K, hidden_size=hidden_size, epoch=N_epochs, num_ens=30, use_parents=use_parents, device=device
    #     # )

    #     print("Running test on counterfactual...")
    #     main_test(Type=Type, mode="counterfactual", parents=parents, data_path=path, save_dir=save_dir, output_dir=output_dir, context_length=context_length,
    #         prediction_length=prediction_length, stride=stride, K=K, hidden_size=hidden_size, epoch=N_epochs, num_ens=30, use_parents=use_parents, device=device
    #     )
    #     N_epochs = 0


    for epoch in range(N_epochs):
        print(f"Start training epoch {epoch+1}")
        node_loss_sum   = { i: 0.0 for i in range(D) }
        node_count      = { i: 0   for i in range(D) }
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for X in loop:
            X = X.to(device)                       # (batch, ctx+pred, D)
            B = X.size(0)
            ctx = X[:, :context_length, :]
            fut = X[:, context_length:, :]
            raw = fut[..., :D]

            opt.zero_grad()

            # always mean-scale
            ctx_raw  = ctx[..., :D]
            norm_ctx, loc, scale = scaler(ctx_raw)

            raw = (raw - loc) / scale
            fut = raw
            ctx_in = norm_ctx

            # per-node forward over ctx_in
            hidden = [None]*D
            h = []
            for i, rn in enumerate(rnn_list):
                inp = ctx_in[..., i:(i+1)]
                _, h_top, hid = rn(inp, hidden[i])
                h.append(h_top)
                hidden[i] = hid

            loss = 0.0
            condition_x_last = norm_ctx[:, -K:, :] if K > 0 else norm_ctx[:, :0, :]

            x_pred_prev = None

            for t in range(prediction_length):
                x_true = raw[:, t:t+1, :]
                step_in_rnn = fut[:, t:t+1, :]

                h_curr = h_next if t > 0 else h
                h_next = []
                for i, rn in enumerate(rnn_list):
                    inp = step_in_rnn[..., i:(i+1)]
                    _, h_i, hid_i = rn(inp, hidden[i])
                    h_next.append(h_i)
                    hidden[i] = hid_i

                for i in range(D):
                    x_i = x_true[:, 0, i].unsqueeze(1)
                    if use_parents:
                        x_parents_hidden_list = [h_curr[j] for j in parents[i]]
                        x_parents_list = condition_x_last[:, -1, parents[i]]
                        if K > 0:
                            cond = torch.cat([h_curr[i], condition_x_last[:,:,i], x_parents_list] + x_parents_hidden_list, dim=1)
                        else:
                            cond = torch.cat([h_curr[i], x_parents_list] + x_parents_hidden_list, dim=1)
                    else:
                        cond = torch.cat([h_curr[i], condition_x_last[:,:,i]], dim=1) if K > 0 else torch.cat([h_curr[i]], dim=1)

                    s   = torch.rand(B, 1, device=device)
                    z   = torch.randn(B, 1, device=device)
                    x_s = (1 - s) * x_i + s * z

                    inp_state = torch.cat([x_s, cond], dim=1)
                    v_pred    = cnfs[i].forward(s, inp_state)
                    v_true = z - x_i
                    this_loss = F.mse_loss(v_pred, v_true)
                    loss  += this_loss
                    node_loss_sum[i] += this_loss.item()
                    node_count[i] += 1

                x_pred_prev = None

                new_step = raw[:, t:t+1, :]
                condition_x_last = torch.cat([condition_x_last[:,1:,:], new_step], dim=1)

            loss.backward()
            clip_grad_value_(
                [p for rn in rnn_list for p in rn.parameters()] + [p for cnf in cnfs for p in cnf.parameters()],
                clip_value=1.0
            )
            opt.step()
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1))

        avg_node_train = { i: (node_loss_sum[i] / node_count[i]) if node_count[i]>0 else float('inf') for i in range(D) }
        avg = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:2d}, avg flow‐matching loss {avg:.6f}")

        if (epoch+1) % 5 == 0:
            for i, rn in enumerate(rnn_list):
                torch.save(rn.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}_rnn_node_{i}.pth"))
            for i, cnf in enumerate(cnfs):
                torch.save(cnf.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}_cnf_node_{i}.pth"))
            print("########################################################")
            print(f"## Completed training epoch {epoch+1}; Starting test ####################")
            print("########################################################")
            print(f"Models saved under {save_dir}")
            print("Running test on observational forecasting...")
            output_dir = f"simulation/results/{Type}_hidden_par_context_{context_length}_prediction_{prediction_length}"
            main_test(Type=Type, mode="forecasting", parents=parents, data_path=path, save_dir=save_dir, output_dir=output_dir, context_length=context_length,
                prediction_length=prediction_length, stride=stride, K=K, hidden_size=hidden_size, epoch=epoch+1, num_ens=30, use_parents=use_parents, device=device
            )

            print("Running test on interventional...")
            main_test(Type=Type, mode="interventional", parents=parents, data_path=path, save_dir=save_dir, output_dir=output_dir, context_length=context_length,
                prediction_length=prediction_length, stride=stride, K=K, hidden_size=hidden_size, epoch=epoch+1, num_ens=30, use_parents=use_parents, device=device
            )

            print("Running test on counterfactual...")
            main_test(Type=Type, mode="counterfactual", parents=parents, data_path=path, save_dir=save_dir, output_dir=output_dir, context_length=context_length,
                prediction_length=prediction_length, stride=stride, K=K, hidden_size=hidden_size, epoch=epoch+1, num_ens=30, use_parents=use_parents, device=device
            )