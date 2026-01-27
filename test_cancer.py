import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def to_t(x, device): return torch.tensor(x, dtype=torch.float32, device=device)

def rollout_one_option(y_ctx_n, A_ctx, y_loc, y_scale, opt_rows,
                       rnn_y, rnn_a, cnf_y, device, K, DA):
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
    y_true = to_t(opt_rows["cancer_volume"].values, device=device).view(5,1)
    y_true_n = (y_true - y_loc) / (y_scale + 1e-8)            # (5,1)

    Capp = to_t(opt_rows["chemo_application_prev"].values, device=device).view(5,1)
    Rapp = to_t(opt_rows["radio_application_prev"].values, device=device).view(5,1)
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
