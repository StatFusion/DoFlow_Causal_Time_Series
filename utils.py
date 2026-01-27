from __future__ import annotations

import torch
import numpy as np
import pandas as pd
import math
import xarray as xr
import os


def simulate_future_interventional(Type, B, D, parents, prediction_length, device, ctx, true_future):
    fut_slice = ctx[:, -((15+1) + prediction_length):-(15+1), :D].clone()  # (B, T, D)
    fut_slice[:,0,0] = ctx[:, -1, 0]
    
    if Type == 'cyclic_linear_high_dimension':
        A, P, phi = 1.0, 20, 0.0
        beta = [0.2] * 50
        beta_self = [0.6]*50
        sigma = [0.2] * 50
        U = [torch.randn(B, prediction_length, device=device) for i in range(50)]
        X = torch.zeros(B, prediction_length, 50, device=device)
        X1_future = ctx[:, -(15 + prediction_length):-15, 0]
        X[:, 0, 1:] = ctx[:, -1, 1:]
        X[:,:,0] = X1_future
        X[:,0,0] = ctx[:, -1, 0]
        eps = [torch.randn(B, prediction_length, device=device) * sigma[i] for i in range(50)]
        # for t in range(1, prediction_length + 1):
        #     for i in range(1, 50):
        #         ar = beta_self[i] * X[:, t-1, i]
        #         infl = beta[i-1]*X[:, t-1, i-1] + 0.7 * X[:, t-1, 0]
        #         X[:, t-1, i] = ar + infl + eps[i][:, t-1] + U[i][:, t-1]
        for t in range(1, prediction_length):
            prev = X[:, t-1, :]
            for i in range(1, 50):
                ar = beta_self[i] * prev[:, i]
                infl = beta[i-1]*prev[:, i-1] + 0.7 * prev[:, 0]
                X[:, t, i] = ar + infl + eps[i][:, t] + U[i][:, t]

        fut_slice = X
        fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)

    elif Type == 'balanced_tail_linear_complicated':
        A, P, phi = 1.0, 20, 0.0
        beta_diag = [0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.2, 0.2]            # β_ii for i=1..8
        sigma     = [0.2] * 8            # noise σ_i for i=1..8
        if Type == 'balanced_tail_linear_beta_2':
            beta_par = {
                (1, 0): 0.3,  # X2 <- X1
                (2, 0): 0.3,  # X3 <- X1
                (3, 1): 0.3,  # X4 <- X2
                (4, 1): 0.3,  # X5 <- X2
                (5, 2): 0.5,  # X6 <- X3
                (6, 2): 0.5,  # X7 <- X3
                (7, 6): 0.5,  # X8 <- X7
            } 
        else:
            beta_par = {
                (1, 0): 0.3,  # X2 <- X1
                (2, 0): -0.3,  # X3 <- X1
                (3, 1): 0.3,  # X4 <- X2
                (4, 1): -0.3,  # X5 <- X2
                (5, 2): 0.5,  # X6 <- X3
                (6, 2): -0.5,  # X7 <- X3
                (7, 6): 0.5,  # X8 <- X7
            } # 2
        X = torch.zeros(B, prediction_length, 8, device=device)
        X1_future = ctx[:, -(15 + prediction_length):-15, 0]
        noise = [torch.randn(B, prediction_length, device=device) * sigma[i] for i in range(8)]
        X[:, 0, 1:] = ctx[:, -1, 1:]
        X[:,:,0] = X1_future
        X[:,0,0] = ctx[:, -1, 0]
        # for t in range(1, prediction_length + 1):
        #     for i in range(1, 8):
        #         ar = beta_diag[i] * X[:, t-1, i]
        #         infl = sum(beta_par[(i, j)] * X[:, t-1, j] for j in parents[i])
        #         X[:, t-1, i] = ar + infl + noise[i][:, t-1]

        for t in range(1, prediction_length):
            prev = X[:, t-1, :]          # previous step (B, D)
            for i in range(1, D):
                ar = beta_diag[i] * prev[:, i]
                infl = sum(beta_par[(i, j)] * prev[:, j] for j in parents[i])  # or your infl rule
                X[:, t, i] = ar + infl + noise[i][:, t]

        fut_slice = X
        fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)
    
    elif Type == 'diamond_3':
        beta_diag = [0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2] 
        beta_par = {
            (1, 0): 0.3,  # X2 <- X1
            (2, 0): 0.3,  # X3 <- X1
            (3, 1): 0.3,  # X4 <- X2
            (5, 1): 0.3,  # X5 <- X2
            (4, 2): 0.5,  # X6 <- X3
            (6, 2): 0.5,  # X7 <- X3
            (7, 3): 0.5,  # X8 <- X4
            (7, 5): 0.5,  # X8 <- X6
            (8, 4): 0.5,  # X9 <- X5
            (8, 6): 0.5,  # X9 <- X7
            (9, 7): 0.5,  # X10 <- X8
            (9, 8): 0.5,  # X10 <- X9
        }
        sigma = [0.20] * 10
        X = torch.zeros(B, prediction_length, D, device=device)
        X1_future = ctx[:, -(15 + prediction_length):-15, 0]
        noise = [torch.randn(B, prediction_length, device=device) * sigma[i] for i in range(D)]
        X[:, 0, 1:] = ctx[:, -1, 1:]
        X[:,:,0] = X1_future
        X[:,0,0] = ctx[:, -1, 0]
        # for t in range(1, prediction_length + 1):
        #     for i in range(1, D):
        #         ar = beta_diag[i] * X[:, t-1, i]
        #         infl = sum(beta_par.get((i, j), 0.0) * X[:, t-1, j]
        #                 for j in range(10))
        #         X[:, t-1, i] = ar + infl + noise[i][:, t-1]

        for t in range(1, prediction_length):
            prev = X[:, t-1, :]
            for i in range(1, D):
                ar = beta_diag[i] * prev[:, i]
                infl = sum(beta_par.get((i, j), 0.0) * prev[:, j] for j in range(10))
                X[:, t, i] = ar + infl + noise[i][:, t]

        fut_slice = X
        fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)
    
    elif Type == 'two_layer_feed_forward_nonlinear':
        phis = np.linspace(0, 2*np.pi, 10)
        As   = 0.5 + np.arange(10)*0.1
        Ps   = 15 + np.arange(10)*2
        beta_diag = [0.5, 0.6, 0.7] + [0.4,0.45,0.5,0.55] + [0.3,0.35,0.4]
        sigma     = [0.3,0.25,0.2] + [0.2,0.25,0.3,0.35] + [0.4,0.35,0.3]
        beta_par = {}
        for hid in range(3,7):
            for inp in range(0,3):
                beta_par[(hid,inp)] = 0.2 + inp/10
        for out in range(7,10):
            for hid in range(3,7):
                beta_par[(out,hid)] = 0.2
        
        def f_exp (x, gamma=0.30): 
            return x + gamma*(torch.exp(x-1)-1) - gamma * torch.tanh(x)

        X = torch.zeros(B, prediction_length, D, device=device)
        noise = [torch.randn(B, prediction_length, device=device) * sigma[i] for i in range(D)]
        X[:, 0, 1:] = ctx[:, -1, 1:]
        X[:,:,0] = ctx[:, -(10 + prediction_length):-10, 0]
        X[:,0,0] = ctx[:, -1, 0]

        # for t in range(1, prediction_length + 1):
        #     for i in range(1,3):
        #         S_t = As[i] * np.sin(2*np.pi*t/Ps[i] + phis[i])
        #         X[:, t-1, i] = beta_diag[i]*X[:, t-1, i] + S_t + noise[i][:, t-1]
        #     for i in range(3, D):
        #         ar = beta_diag[i] * X[:, t-1, i]
        #         infl = sum(beta_par[(i, j)] * X[:, t-1, j] for j in parents[i])
        #         X[:, t-1, i] = ar + f_exp(infl) + noise[i][:, t-1]

        for t in range(1, prediction_length):
            prev = X[:, t-1, :]
            for i in range(1, 3):
                S_t = As[i] * np.sin(2*np.pi*t/Ps[i] + phis[i])
                X[:, t, i] = beta_diag[i]*prev[:, i] + S_t + noise[i][:, t]
            for i in range(3, D):
                ar = beta_diag[i] * prev[:, i]
                infl = sum(beta_par[(i, j)] * prev[:, j] for j in parents[i])
                X[:, t, i] = ar + f_exp(infl) + noise[i][:, t]

        fut_slice = X
        fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)
    return fut



def simulate_future_counterfactual(
    Type: str,
    ctx_history: torch.Tensor,       # (B, history_length, D)
    parents: dict[int, list[int]],   # adjacency info
    prediction_length: int,
    device: torch.device,
    true_future: torch.Tensor,        # (B, H, total_dims)
    inter_set: list[int]
) -> torch.Tensor:

    B, H, D = ctx_history.shape[0], prediction_length, ctx_history.shape[2]
    total_dims = true_future.size(2)
    noise = torch.zeros(B, H, D, device=device)

    X_intervention = ctx_history[:, -(15 + prediction_length):-15, inter_set]

    
    if Type == "cyclic_linear_high_dimension":
        A, P, phi = 1.0, 20, 0.0
        beta = [0.2] * 50
        beta_self = [0.6]*50
        sigma = [0.2] * 50
        
        for t in range(H):
            for j in range(1, D):
                prev_val = true_future[:, t-1, j] if t>0 else ctx_history[:, -1, j]
                ar = beta_self[j] * prev_val
                infl = beta[j-1]*true_future[:, t-1, j-1] + 0.7 * true_future[:, t-1, 0]
                noise[:, t, j] = true_future[:, t, j] - (ar + infl)

    
    elif Type == "balanced_tail_linear_complicated":
        beta_diag = [0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.2, 0.2]
        beta_par = {
            (1, 0): 0.3,  # X2 <- X1
            (2, 0): -0.3,  # X3 <- X1
            (3, 1): 0.3,  # X4 <- X2
            (4, 1): -0.3,  # X5 <- X2
            (5, 2): 0.5,  # X6 <- X3
            (6, 2): -0.5,  # X7 <- X3
            (7, 6): 0.5,  # X8 <- X7
        }
        for t in range(H):
            for j in range(1, D):
                prev_val = true_future[:, t-1, j] if t>0 else ctx_history[:, -1, j]
                ar = beta_diag[j] * prev_val
                infl = sum(
                    beta_par[(j, k)] *
                    (true_future[:, t-1, k] if t>0 else ctx_history[:, -1, k])
                    for k in parents[j]
                )
                noise[:, t, j] = true_future[:, t, j] - (ar + infl)
    

    elif Type == "diamond_3":
        beta_diag = [0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2]
        beta_par = {
            (1, 0): 0.3,  # X2 <- X1
            (2, 0): 0.3,  # X3 <- X1
            (3, 1): 0.3,  # X4 <- X2
            (5, 1): 0.3,  # X5 <- X2
            (4, 2): 0.5,  # X6 <- X3
            (6, 2): 0.5,  # X7 <- X3
            (7, 3): 0.5,  # X8 <- X4
            (7, 5): 0.5,  # X8 <- X6
            (8, 4): 0.5,  # X9 <- X5
            (8, 6): 0.5,  # X9 <- X7
            (9, 7): 0.5,  # X10 <- X8
            (9, 8): 0.5,  # X10 <- X9
        } 
        for t in range(H):
            for j in range(1, D):
                prev_val = true_future[:, t-1, j] if t>0 else ctx_history[:, -1, j]
                ar = beta_diag[j] * prev_val
                infl = sum(
                    beta_par[(j, k)] *
                    (true_future[:, t-1, k] if t>0 else ctx_history[:, -1, k])
                    for k in parents[j]
                )
                noise[:, t, j] = true_future[:, t, j] - (ar + infl)
    
    elif Type == "two_layer_feed_forward_nonlinear":
        phis = np.linspace(0, 2*np.pi, 10)
        As   = 0.5 + np.arange(10)*0.1
        Ps   = 15 + np.arange(10)*2
        beta_diag = [0.5, 0.6, 0.7] + [0.4,0.45,0.5,0.55] + [0.3,0.35,0.4]
        sigma = [0.3,0.25,0.2] + [0.2,0.25,0.3,0.35] + [0.4,0.35,0.3]
        beta_par = {}
        for hid in range(3,7):
            for inp in range(0,3):
                beta_par[(hid,inp)] = 0.2 + inp/10
        for out in range(7,10):
            for hid in range(3,7):
                beta_par[(out,hid)] = 0.2

        def f_exp (x, gamma=0.30): 
            return x + gamma*(torch.exp(x-1)-1) - gamma * torch.tanh(x)
        


        for t in range(H):
            for j in range(3, D):
                prev_val = true_future[:, t-1, j] if t>0 else ctx_history[:, -1, j]
                ar = beta_diag[j] * prev_val
                infl = sum(
                    beta_par[(j, k)] *
                    (true_future[:, t-1, k] if t>0 else ctx_history[:, -1, k])
                    for k in parents[j]
                )
                noise[:, t, j] = true_future[:, t, j] - (ar + f_exp(infl))

    else:
        raise ValueError(f"Unknown Type {Type}")

    cf = torch.zeros(B, H, D, device=device)
    cf[:, 0, :] = ctx_history[:, -1, :]
    cf[:, :, inter_set] = X_intervention

    # for t in range(1, H + 1):
    for t in range(H):
        if t == 0:
            prev = ctx_history[:, -1, :]
        else:
            prev = cf[:, t-1, :]
            
        if Type == "cyclic_linear_high_dimension":
            x0 = prev[:, 0]  # (B,)
            for j in range(1, D):
                ar = beta_self[j] * prev[:, j]
                infl = beta[j-1] * prev[:, j-1] + 0.7 * x0
                cf[:, t, j] = ar + infl + noise[:, t, j]

        elif Type == "balanced_tail_linear_complicated":
            for j in range(1, D):
                ar_term = beta_diag[j] * prev[:, j]
                infl_term = sum(
                    beta_par[(j, k)] * prev[:, k]
                    for k in parents[j]
                )
                cf[:, t, j] = ar_term + infl_term + noise[:, t, j]
        
        elif Type == "diamond_3":
            for j in range(1, D):
                ar_term = beta_diag[j] * prev[:, j]
                infl_term = sum(
                    beta_par[(j, k)] * prev[:, k]
                    for k in parents[j]
                )
                cf[:, t, j] = ar_term + infl_term + noise[:, t, j]
        
        elif Type == "two_layer_feed_forward_nonlinear":
            for j in range(3, D):
                ar_term = beta_diag[j] * prev[:, j]
                infl_term = sum(
                    beta_par[(j, k)] * prev[:, k]
                    for k in parents[j]
                )
                cf[:, t, j] = ar_term + f_exp(infl_term) + noise[:, t, j]

        cf[:, t, inter_set] = X_intervention[:, t, :]

    if total_dims > D:
        cf_full = torch.cat([cf, true_future[:, :, D:]], dim=2)
    else:
        cf_full = cf
    return cf_full


# def quantiles_and_logp(logp_list, samples_arr, prediction_length, B, num_ens, D, mean_scalar=True, loc=None, scale=None):
#     if logp_list is not None:
#         logp_arr = torch.stack(logp_list, axis=0).cpu().numpy()      # → (T, E, D)
#         logp_arr = logp_arr.reshape(prediction_length, B, num_ens, D)

#         logp_sum_over_time = logp_arr.sum(axis=0)  # (B, num_ens, D)
#         best_chain_idx = logp_sum_over_time.sum(axis=-1).argmax(axis=-1)  # (B,)
#         batch_idx = np.arange(B)
#         best_samples = samples_arr[:, batch_idx, best_chain_idx, :]  # (prediction_length, B, D)

#         logp_chain = logp_sum_over_time.sum(axis=-1)  # (B, num_ens)
#         logp_avg = logp_arr.mean(axis=2)           # → (T, B, D)
#         logp_sum = logp_avg.sum(axis=0)

#         logp_mean = logp_sum.mean(axis=0)  # → (D,)
#         logp_std = logp_sum.std(axis=0)    # → (D,)
#     else:
#         logp_sum, logp_mean, logp_std = None, None, None

#     # compute quantiles
#     q5 = np.nanquantile(samples_arr, 0.05, axis=2)
#     q10 = np.nanquantile(samples_arr, 0.10, axis=2)
#     q25 = np.nanquantile(samples_arr, 0.25, axis=2)
#     q50 = np.nanquantile(samples_arr, 0.50, axis=2)
#     q75 = np.nanquantile(samples_arr, 0.75, axis=2)
#     q90 = np.nanquantile(samples_arr, 0.90, axis=2)
#     q95 = np.nanquantile(samples_arr, 0.95, axis=2)
#     if mean_scalar:
#         loc_np   = loc.cpu().numpy().squeeze(1)    # → (3,8)
#         scale_np = scale.cpu().numpy().squeeze(1)
#         q5 = q5 * scale_np[None, :, :] + loc_np[None, :, :]
#         q10 = q10 * scale_np[None, :, :] + loc_np[None, :, :]
#         q25 = q25 * scale_np[None, :, :] + loc_np[None, :, :]
#         q50 = q50 * scale_np[None, :, :] + loc_np[None, :, :]
#         q75 = q75 * scale_np[None, :, :] + loc_np[None, :, :]
#         q90 = q90 * scale_np[None, :, :] + loc_np[None, :, :]
#         q95 = q95 * scale_np[None, :, :] + loc_np[None, :, :]
#     return logp_sum, logp_mean, logp_std, q5, q10, q25, q50, q75, q90, q95


def quantiles_and_logp(
    logp_list,                # list of T tensors (B*E, D, 1) or None
    samples_arr_t,            # torch.Tensor (T, B, E, D) on *GPU*
    device,
    loc=None, scale=None      # torch.Tensors (B, 1, D) if mean_scalar
):
    if isinstance(samples_arr_t, np.ndarray):
        samples_arr_t = torch.from_numpy(samples_arr_t).to(device=device, dtype=torch.float32)
    else:
        samples_arr_t = samples_arr_t.to(device=device, dtype=torch.float32)

    if logp_list is not None:
        logp_t = torch.stack(logp_list, dim=0)
        if logp_t.dim() == 4 and logp_t.size(-1) == 1:
            logp_t = logp_t.squeeze(-1)         # (T, B*E, D)
        T, BE, D_ = logp_t.shape
        _, B, E, Dchk = samples_arr_t.shape
        assert D_ == Dchk, "D mismatch between logp_list and samples_arr"
        logp_t = logp_t.view(T, B, E, D_)       # (T,B,E,D)
        logp_avg  = logp_t.mean(dim=2)          # (T,B,D)
        logp_sum  = logp_avg.sum(dim=0)         # (B,D)
        logp_mean = logp_sum.mean(dim=0)        # (D,)
        logp_std  = logp_sum.std(dim=0)         # (D,)
    else:
        logp_sum = logp_mean = logp_std = None

    # ---- quantiles over ensemble dim=2 (GPU) ----
    qs = torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
                      device=samples_arr_t.device, dtype=samples_arr_t.dtype)
    q_all = torch.nanquantile(samples_arr_t, qs, dim=2)
    loc   = loc.to(device=samples_arr_t.device, dtype=samples_arr_t.dtype)
    scale = scale.to(device=samples_arr_t.device, dtype=samples_arr_t.dtype)
    loc_b   = loc.unsqueeze(0).unsqueeze(1)    # (1,1,B,1,D)
    scale_b = scale.unsqueeze(0).unsqueeze(1)  # (1,1,B,1,D)
    q_all = (q_all.unsqueeze(3) * scale_b + loc_b).squeeze(3)
    q5, q10, q25, q50, q75, q90, q95 = [q_all[i] for i in range(7)]
    return logp_sum, logp_mean, logp_std, q5, q10, q25, q50, q75, q90, q95



def metrics(fut, samples_arr, device, loc, scale, output_dir, mode):
    T, B, E, D = samples_arr.shape
    if isinstance(samples_arr, np.ndarray):
        samples_arr = torch.from_numpy(samples_arr).to(device)
    else:
        samples_arr = samples_arr.to(device=device, dtype=torch.float32)

    fut_raw    = fut[..., :D]                       # (B, T, D)
    fut_scaled = (fut_raw - loc) / scale            # (B, T, D)
    truth      = fut_scaled.transpose(0, 1)         # (T, B, D)

    forecast_mean = torch.nanmean(samples_arr, dim=2)  # (T, B, D)

    sq_err       = (forecast_mean - truth) ** 2        # (T, B, D)
    rmse_t_d     = torch.sqrt(sq_err.mean(dim=1))      # (T, D)
    rmse_chain_d = torch.sqrt(sq_err.mean(dim=(0, 1))) # (D,)
    np.savetxt(
        os.path.join(output_dir, f"rmse_{mode}.csv"),
        rmse_t_d.detach().cpu().numpy(),
        delimiter=","
    )
    np.savetxt(
        os.path.join(output_dir, f"rmse_one_chain_{mode}.csv"),
        rmse_chain_d.detach().cpu().numpy()[None, :],
        delimiter=","
    )
    print(f"Average RMSE across chains: {rmse_chain_d.mean().item():.4f}")
    return
