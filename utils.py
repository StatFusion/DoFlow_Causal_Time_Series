import torch
import numpy as np
import pandas as pd
import math
import xarray as xr
from scores.probability import crps_for_ensemble
import os


def simulate_future_interventional(Type, B, D, parents, prediction_length, device, ctx, true_future):
    fut_slice = ctx[:, -((15+1) + prediction_length):-(15+1), :D].clone()  # (B, T, D)
    fut_slice[:,0,0] = ctx[:, -1, 0]
    if Type == 'cyclic':
            X1_future = ctx[:, -(15 + prediction_length):-15, 0]
            A, P, phi = 1.0, 30, 0.0
            beta21, beta33 = 0.5, 0.5
            sigma2, sigma3 = 0.2, 0.2
            eps2 = torch.randn(B, prediction_length, device=device) * sigma2
            eps3 = torch.randn(B, prediction_length, device=device) * sigma3
            U2   = torch.randn(B, prediction_length, device=device)
            U3   = torch.randn(B, prediction_length, device=device)
            X2 = torch.zeros(B, prediction_length, device=device)
            X3 = torch.zeros(B, prediction_length, device=device)
            X2[:, 0] = ctx[:, -1, 1]
            X3[:, 0] = ctx[:, -1, 2] 
            for t in range(1, prediction_length + 1):
                X2[:, t-1] = (
                    torch.exp(fut_slice[:, t-1, 0] / 5)
                    + U2[:, t-1] / 4
                    + beta21 * X2[:, t - 1]
                    + eps2[:, t-1]
                )
                X3[:, t-1] = (
                    (X2[:, t - 1] - 3) ** 2
                    + U3[:, t-1]
                    + beta33 * X3[:, t - 1]
                    + eps3[:, t-1]
                )
            fut_slice[:, :, 0] = X1_future
            fut_slice[:, :, 1] = X2
            fut_slice[:, :, 2] = X3
            fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)

    elif Type == 'cyclic_linear_high_dimension':
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
            for t in range(1, prediction_length + 1):
                for i in range(1, 50):
                    ar = beta_self[i] * X[:, t-1, i]
                    infl = beta[i-1]*X[:, t-1, i-1] + 0.7 * X[:, t-1, 0]
                    X[:, t-1, i] = ar + infl + eps[i][:, t-1] + U[i][:, t-1]
            fut_slice = X
            fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)


    elif Type == 'balanced_tail_linear':
        A, P, phi = 1.0, 20, 0.0
        beta_diag = [0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.7, 0.8]            # β_ii for i=1..8
        sigma     = [0.2] * 8            # noise σ_i for i=1..8
        beta_par = {
            (1, 0): 0.3,  # X2 <- X1
            (2, 0): 0.3,  # X3 <- X1
            (3, 1): 0.3,  # X4 <- X2
            (4, 1): 0.3,  # X5 <- X2
            (5, 2): 0.3,  # X6 <- X3
            (6, 2): 0.3,  # X7 <- X3
            (7, 6): 0.3,  # X8 <- X7
        }
        X = torch.zeros(B, prediction_length, 8, device=device)
        noise = [torch.randn(B, prediction_length, device=device) * sigma[i] for i in range(8)]
        X1_future = ctx[:, -(15 + prediction_length):-15, 0]
        X[:, 0, 1:] = ctx[:, -1, 1:]
        X[:,:,0] = X1_future
        X[:,0,0] = ctx[:, -1, 0]
        for t in range(1, prediction_length + 1):
            for i in range(1, 8):
                ar = beta_diag[i] * X[:, t-1, i]
                infl = sum(beta_par[(i, j)] * X[:, t-1, j] for j in parents[i])
                X[:, t-1, i] = ar + infl + noise[i][:, t-1]
        fut_slice = X
        fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)

    elif Type == 'balanced_tail_linear_beta_2' or Type == 'balanced_tail_linear_complicated':
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
        for t in range(1, prediction_length + 1):
            for i in range(1, 8):
                ar = beta_diag[i] * X[:, t-1, i]
                infl = sum(beta_par[(i, j)] * X[:, t-1, j] for j in parents[i])
                X[:, t-1, i] = ar + infl + noise[i][:, t-1]
        fut_slice = X
        fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)
    
    elif Type == 'balanced_tail_non_additive_beta_2':
        A, P, phi = 1.0, 20, 0.0
        beta_diag = [0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.2, 0.2]            # β_ii for i=1..8
        sigma     = [0.3] * 8            # noise σ_i for i=1..8
        beta_par = {
            (1, 0): 0.3,  # X2 <- X1
            (2, 0): 0.3,  # X3 <- X1
            (3, 1): 0.3,  # X4 <- X2
            (4, 1): 0.3,  # X5 <- X2
            (5, 2): 0.5,  # X6 <- X3
            (6, 2): 0.5,  # X7 <- X3
            (7, 6): 0.5,  # X8 <- X7
        } # 2
        X = torch.zeros(B, prediction_length, 8, device=device)
        X1_future = ctx[:, -(15 + prediction_length):-15, 0]
        noise = [torch.randn(B, prediction_length, device=device) * sigma[i] for i in range(8)]
        X[:, 0, 1:] = ctx[:, -1, 1:]
        X[:,:,0] = X1_future
        X[:,0,0] = ctx[:, -1, 0]
        for t in range(1, prediction_length + 1):
            for i in range(1, 8):
                ar = beta_diag[i] * X[:, t-1, i]
                infl = sum(beta_par[(i, j)] * X[:, t-1, j] for j in parents[i])
                X[:, t-1, i] = ar * (torch.abs(noise[i][:, t-1]) + 0.5) + infl
        fut_slice = X
        fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)
    
    elif Type == 'balanced_tail_nonlinear':
        A, P, phi = 1.0, 20, 0.0
        beta_diag = [0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.2, 0.2]            # β_ii for i=1..8
        sigma     = [0.2] * 8            # noise σ_i for i=1..8
        beta_par = {
            (1, 0): 0.3,  # X2 <- X1
            (2, 0): 0.3,  # X3 <- X1
            (3, 1): 0.3,  # X4 <- X2
            (4, 1): 0.3,  # X5 <- X2
            (5, 2): 0.5,  # X6 <- X3
            (6, 2): 0.5,  # X7 <- X3
            (7, 6): 0.5,  # X8 <- X7
        }
        nonlin_type = {
            1: None,
            2: 'exp',   # X3 uses exp
            3: 'tanh',  # X4 uses tanh
            4: None,
            5: 'exp',
            6: None,
            7: 'tanh',
        }
        X = torch.zeros(B, prediction_length, 8, device=device)
        X1_future = ctx[:, -(15 + prediction_length):-15, 0]
        noise = [torch.randn(B, prediction_length, device=device) * sigma[i] for i in range(8)]
        X[:, 0, 1:] = ctx[:, -1, 1:]
        X[:,:,0] = X1_future
        X[:,0,0] = ctx[:, -1, 0]
        for t in range(1, prediction_length + 1):
            for i in range(1, 8):
                ar_term = beta_diag[i] * X[:, t-1, i]
                lin_infl = sum(beta_par[(i, j)] * X[:, t-1, j] for j in parents[i])
                kind = nonlin_type[i]
                if kind == 'exp':
                    infl = torch.exp(lin_infl / 5.0) - 1.0
                elif kind == 'tanh':
                    infl = torch.tanh(lin_infl / 2.0) * lin_infl
                else:
                    infl = lin_infl
                X[:, t-1, i] = ar_term + infl + noise[i][:, t-1]
        fut_slice = X
        fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)
    
    elif Type == 'diamond':
        beta_diag = [0.6, 0.5, 0.55, 0.45, 0.5, 
             0.4, 0.35, 0.3,  0.25, 0.2]
        beta_par = {
            (1, 0): 0.30,   # X2 ← X1
            (2, 0): 0.35,   # X3 ← X1
            (3, 1): 0.25,   # X4 ← X2
            (5, 1): 0.40,   # X6 ← X2
            (4, 2): 0.30,   # X5 ← X3
            (6, 2): 0.45,   # X7 ← X3
            (7, 3): 0.50,   # X8 ← X4
            (7, 5): 0.20,   # X8 ← X6
            (8, 4): 0.40,   # X9 ← X5
            (8, 6): 0.30,   # X9 ← X7
            (9, 7): 0.60,   # X10 ← X8
            (9, 8): 0.20,   # X10 ← X9
        }
        sigma = [0.20, 0.25, 0.22, 0.30, 0.28, 
                0.24, 0.26, 0.20, 0.30, 0.35]
        X = torch.zeros(B, prediction_length, D, device=device)
        X1_future = ctx[:, -(4 + prediction_length):-4, 0]
        noise = [torch.randn(B, prediction_length, device=device) * sigma[i] for i in range(D)]
        X[:, 0, 1:] = ctx[:, -1, 1:]
        X[:,:,0] = X1_future
        X[:,0,0] = ctx[:, -1, 0]
        for t in range(1, prediction_length + 1):
            for i in range(1, D):
                ar = beta_diag[i] * X[:, t-1, i]
                infl = sum(beta_par.get((i, j), 0.0) * X[:, t-1, j]
                        for j in range(10))
                X[:, t-1, i] = ar + infl + noise[i][:, t-1]
        fut_slice = X
        fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)
    
    elif Type == 'diamond_2':
        beta_diag = [0.6, 0.5, 0.55, 0.45, 0.5, 
             0.4, 0.35, 0.3,  0.25, 0.2]
        beta_par = {
            # Level0→1
            (1, 0): 0.30,   # X2 ← X1
            (2, 0): 0.30,   # X3 ← X1

            # Level1→2
            (3, 1): 0.30,   # X4 ← X2
            (5, 1): 0.30,   # X6 ← X2
            (4, 2): 0.30,   # X5 ← X3
            (6, 2): 0.30,   # X7 ← X3

            # Level2→3
            (7, 3): 0.40,   # X8 ← X4
            (7, 5): 0.40,   # X8 ← X6
            (8, 4): 0.40,   # X9 ← X5
            (8, 6): 0.40,   # X9 ← X7

            # Level3→4
            (9, 7): 0.50,   # X10 ← X8
            (9, 8): 0.50,   # X10 ← X9
        }
        sigma = [0.20] * 10
        X = torch.zeros(B, prediction_length, D, device=device)
        X1_future = ctx[:, -(15 + prediction_length):-15, 0]
        noise = [torch.randn(B, prediction_length, device=device) * sigma[i] for i in range(D)]
        X[:, 0, 1:] = ctx[:, -1, 1:]
        X[:,:,0] = X1_future
        X[:,0,0] = ctx[:, -1, 0]
        for t in range(1, prediction_length + 1):
            for i in range(1, D):
                ar = beta_diag[i] * X[:, t-1, i]
                infl = sum(beta_par.get((i, j), 0.0) * X[:, t-1, j]
                        for j in range(10))
                X[:, t-1, i] = ar + infl + noise[i][:, t-1]
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
        for t in range(1, prediction_length + 1):
            for i in range(1, D):
                ar = beta_diag[i] * X[:, t-1, i]
                infl = sum(beta_par.get((i, j), 0.0) * X[:, t-1, j]
                        for j in range(10))
                X[:, t-1, i] = ar + infl + noise[i][:, t-1]
        fut_slice = X
        fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)
    
    elif Type == 'diamond_3_non_additive':
        beta_diag = [0.5] * 10 
        beta_par = {
            (1, 0): 1.0,  # X2 <- X1
            (2, 0): 1.0,  # X3 <- X1
            (3, 1): 0.4,  # X4 <- X2
            (5, 1): 0.4,  # X5 <- X2
            (4, 2): 0.4,  # X6 <- X3
            (6, 2): 0.4,  # X7 <- X3
            (7, 3): 0.4,  # X8 <- X4
            (7, 5): 0.4,  # X8 <- X6
            (8, 4): 0.4,  # X9 <- X5
            (8, 6): 0.4,  # X9 <- X7
            (9, 7): 0.4,  # X10 <- X8
            (9, 8): 0.4,  # X10 <- X9
        } 
        sigma = [0.2] * 10
        X = torch.zeros(B, prediction_length, D, device=device)
        X1_future = ctx[:, -(15 + prediction_length):-15, 0]
        noise = [torch.randn(B, prediction_length, device=device) * sigma[i] for i in range(D)]
        X[:, 0, 1:] = ctx[:, -1, 1:]
        X[:,:,0] = X1_future
        X[:,0,0] = ctx[:, -1, 0]
        for t in range(1, prediction_length + 1):
            for i in range(1, D):
                ar = beta_diag[i] * X[:, t-1, i]
                infl = sum(beta_par.get((i, j), 0.0) * X[:, t-1, j]
                        for j in range(10))
                X[:, t-1, i] = 1/(2 + 5 * torch.abs(noise[i][:, t-1])) * torch.exp(ar/3) + infl
        fut_slice = X
        fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)
    
    elif Type == 'diamond_2_square':
        beta_diag = [0.6, 0.5, 0.55, 0.45, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2]
        beta_par = {
            (1,0):0.30, (2,0):0.30,
            (3,1):0.30, (5,1):0.30, (4,2):0.30, (6,2):0.30,
            (7,3):0.40,(7,5):0.40,(8,4):0.40,(8,6):0.40,
            (9,7):0.50,(9,8):0.50
        }
        def f_square(x, gamma=0.10):
            return x + gamma*(x**2)
        sigma = [0.20]*10
        X = torch.zeros(B, prediction_length, D, device=device)
        X1_future = ctx[:, -(4 + prediction_length):-4, 0]
        noise = [torch.randn(B, prediction_length, device=device) * sigma[i] for i in range(D)]
        X[:, 0, 1:] = ctx[:, -1, 1:]
        X[:,:,0] = X1_future
        X[:,0,0] = ctx[:, -1, 0]
        for t in range(1, prediction_length + 1):
            for i in range(1, D):
                ar = beta_diag[i] * X[:, t-1, i]
                infl = sum(beta_par.get((i, j), 0.0) * X[:, t-1, j]
                        for j in range(10))
                lin = ar + infl
                X[:, t-1, i] = f_square(lin) + noise[i][:, t-1]
        fut_slice = X
        fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)
    
    elif Type == 'two_layer_feed_forward':
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
                beta_par[(out,hid)] = 0.2 + (hid-3)/10

        X = torch.zeros(B, prediction_length, D, device=device)
        noise = [torch.randn(B, prediction_length, device=device) * sigma[i] for i in range(D)]
        X[:, 0, 1:] = ctx[:, -1, 1:]
        X[:,:,0] = ctx[:, -(15 + prediction_length):-15, 0]
        X[:,0,0] = ctx[:, -1, 0]

        for t in range(1, prediction_length + 1):
            for i in range(1,3):
                S_t = As[i] * np.sin(2*np.pi*t/Ps[i] + phis[i])
                X[:, t-1, i] = beta_diag[i]*X[:, t-1, i] + S_t + noise[i][:, t-1]
            for i in range(3, D):
                ar = beta_diag[i] * X[:, t-1, i]
                infl = sum(beta_par[(i, j)] * X[:, t-1, j] for j in parents[i])
                X[:, t-1, i] = ar + infl + noise[i][:, t-1]
        fut_slice = X
        fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)
    
    elif Type == 'two_layer_feed_forward_2':
        phis = np.linspace(0, 2*np.pi, 10)
        As   = 0.5 + np.arange(10)*0.1
        Ps   = 15 + np.arange(10)*2
        beta_diag = [0.5, 0.4, 0.3] + [0.2] * 7
        sigma     = [0.2] * 10
        beta_par = {}
        for hid in range(3,7):
            for inp in range(0,3):
                beta_par[(hid,inp)] = 0.3
        for out in range(7,10):
            for hid in range(3,7):
                beta_par[(out,hid)] = 0.5

        X = torch.zeros(B, prediction_length, D, device=device)
        noise = [torch.randn(B, prediction_length, device=device) * sigma[i] for i in range(D)]
        X[:, 0, 1:] = ctx[:, -1, 1:]
        X[:,:,0] = ctx[:, -(15 + prediction_length):-15, 0]
        X[:,0,0] = ctx[:, -1, 0]

        for t in range(1, prediction_length + 1):
            for i in range(1,3):
                S_t = As[i] * np.sin(2*np.pi*t/Ps[i] + phis[i])
                X[:, t-1, i] = beta_diag[i]*X[:, t-1, i] + S_t + noise[i][:, t-1]
            for i in range(3, D):
                ar = beta_diag[i] * X[:, t-1, i]
                infl = sum(beta_par[(i, j)] * X[:, t-1, j] for j in parents[i])
                X[:, t-1, i] = ar + infl + noise[i][:, t-1]
        fut_slice = X
        fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)
    
    elif Type == 'two_layer_feed_forward_non_additive':
        phis = np.linspace(0, 2*np.pi, 10)
        As   = 0.5 + np.arange(10)*0.1
        Ps   = 15 + np.arange(10)*2
        beta_diag = [0.5] * 10
        sigma     = [0.1] * 10
        beta_par = {}
        for hid in range(3,7):
            for inp in range(0,3):
                beta_par[(hid,inp)] = 0.2
        for out in range(7,10):
            for hid in range(3,7):
                beta_par[(out,hid)] = 0.2

        X = torch.zeros(B, prediction_length, D, device=device)
        noise = [torch.randn(B, prediction_length, device=device) * sigma[i] for i in range(D)]
        X[:, 0, 3:] = ctx[:, -1, 3:]
        X[:,:,0:3] = ctx[:, -(10 + prediction_length):-10, 0:3]
        X[:,0,0:3] = ctx[:, -1, 0:3]

        for t in range(1, prediction_length + 1):
            # for i in range(1,3):
            #     S_t = As[i] * np.sin(2*np.pi*t/Ps[i] + phis[i])
            #     X[:, t-1, i] = beta_diag[i]*X[:, t-1, i] + S_t + noise[i][:, t-1]
            for i in range(3, D):
                ar = beta_diag[i] * X[:, t-1, i]
                infl = sum(beta_par[(i, j)] * X[:, t-1, j] for j in parents[i])
                X[:, t-1, i] = (0.5 * torch.abs(infl) + torch.abs(noise[i][:, t-1]))**0.5 + ar
        fut_slice = X

        fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)
    
    elif Type == 'two_layer_feed_forward_non_additive_2':
        phis = np.linspace(0, 2*np.pi, 10)
        As   = 0.5 + np.arange(10)*0.1
        Ps   = 15 + np.arange(10)*2
        beta_diag = [0.5] * 10
        sigma     = [0.1] * 10
        beta_par = {}
        for hid in range(3,7):
            for inp in range(0,3):
                beta_par[(hid,inp)] = 0.2
        for out in range(7,10):
            for hid in range(3,7):
                beta_par[(out,hid)] = 0.2

        X = torch.zeros(B, prediction_length, D, device=device)
        noise = [torch.randn(B, prediction_length, device=device) * sigma[i] for i in range(D)]
        X[:, 0, 1:] = ctx[:, -1, 1:]
        X[:,:,0] = ctx[:, -(15 + prediction_length):-15, 0]
        X[:,0,0] = ctx[:, -1, 0]

        for t in range(1, prediction_length + 1):
            for i in range(1,3):
                S_t = As[i] * np.sin(2*np.pi*t/Ps[i] + phis[i])
                X[:, t-1, i] = beta_diag[i]*X[:, t-1, i] + S_t + noise[i][:, t-1]
            for i in range(3, D):
                ar = beta_diag[i] * X[:, t-1, i]
                infl = sum(beta_par[(i, j)] * X[:, t-1, j] for j in parents[i])
                X[:, t-1, i] = (0.5 * torch.abs(infl) + torch.abs(noise[i][:, t-1]))**2 + ar
        fut_slice = X

        fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)
    
    elif Type == 'two_layer_feed_forward_non_additive_4' or Type == 'two_layer_feed_forward_non_additive_3':
        phis = np.linspace(0, 2*np.pi, 10)
        As   = 0.5 + np.arange(10)*0.1
        Ps   = 15 + np.arange(10)*2
        beta_diag = [0.5] * 10
        sigma     = [0.2] * 10
        beta_par = {}
        for hid in range(3,7):
            for inp in range(0,3):
                beta_par[(hid,inp)] = 0.2
        for out in range(7,10):
            for hid in range(3,7):
                beta_par[(out,hid)] = 0.2

        X = torch.zeros(B, prediction_length, D, device=device)
        noise = [torch.randn(B, prediction_length, device=device) * sigma[i] for i in range(D)]
        X[:, 0, 1:] = ctx[:, -1, 1:]
        X[:,:,0] = ctx[:, -(15 + prediction_length):-15, 0]
        X[:,0,0] = ctx[:, -1, 0]

        for t in range(1, prediction_length + 1):
            for i in range(1,3):
                S_t = As[i] * np.sin(2*np.pi*t/Ps[i] + phis[i])
                X[:, t-1, i] = beta_diag[i]*X[:, t-1, i] + S_t + noise[i][:, t-1]
            for i in range(3, D):
                ar = beta_diag[i] * X[:, t-1, i]
                infl = sum(beta_par[(i, j)] * X[:, t-1, j] for j in parents[i])
                if Type == 'two_layer_feed_forward_non_additive_4':
                    X[:, t-1, i] = (infl/2 + noise[i][:, t-1])* torch.exp(infl/5) + ar
                else:
                    X[:, t-1, i] = (infl/2 + torch.abs(noise[i][:, t-1]))* torch.exp(infl/5) + ar
        fut_slice = X
        fut = torch.cat([fut_slice, true_future[:, :, D:]], dim=2)
    
    elif Type == 'two_layer_feed_forward_non_additive_5':
        phis = np.linspace(0, 2*np.pi, 10)
        As   = 0.5 + np.arange(10)*0.1
        Ps   = 15 + np.arange(10)*2
        beta_diag = [0.5, 0.4, 0.3] + [0.2] * 7
        sigma     = [0.15] * 10

        beta_par = {}
        for hid in range(3,7):
            for inp in range(0,3):
                beta_par[(hid,inp)] = 0.3
        for out in range(7,10):
            for hid in range(3,7):
                beta_par[(out,hid)] = 0.5

        X = torch.zeros(B, prediction_length, D, device=device)
        noise = [torch.randn(B, prediction_length, device=device) * sigma[i] for i in range(D)]
        X[:, 0, 1:] = ctx[:, -1, 1:]
        X[:,:,0] = ctx[:, -(15 + prediction_length):-15, 0]
        X[:,0,0] = ctx[:, -1, 0]

        for t in range(1, prediction_length + 1):
            for i in range(1,3):
                S_t = As[i] * np.sin(2*np.pi*t/Ps[i] + phis[i])
                X[:, t-1, i] = beta_diag[i]*X[:, t-1, i] + S_t + noise[i][:, t-1]
            for i in range(3, D):
                ar = beta_diag[i] * X[:, t-1, i]
                infl = sum(beta_par[(i, j)] * X[:, t-1, j] for j in parents[i])
                X[:, t-1, i] = (infl/2 + noise[i][:, t-1])* torch.exp(infl/5) + ar
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
        X[:,:,0] = ctx[:, -(15 + prediction_length):-15, 0]
        X[:,0,0] = ctx[:, -1, 0]

        for t in range(1, prediction_length + 1):
            for i in range(1,3):
                S_t = As[i] * np.sin(2*np.pi*t/Ps[i] + phis[i])
                X[:, t-1, i] = beta_diag[i]*X[:, t-1, i] + S_t + noise[i][:, t-1]
            for i in range(3, D):
                ar = beta_diag[i] * X[:, t-1, i]
                infl = sum(beta_par[(i, j)] * X[:, t-1, j] for j in parents[i])
                X[:, t-1, i] = ar + f_exp(infl) + noise[i][:, t-1]
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

    if Type == "cyclic":
        for t in range(H):
            X1_prev = true_future[:, t-1, 0] if t>0 else ctx_history[:, -1, 0]
            X2_prev = true_future[:, t-1, 1] if t>0 else ctx_history[:, -1, 1]
            mu2 = torch.exp(X1_prev/5) + 0.5 * X2_prev
            noise[:, t, 1] = true_future[:, t, 1] - mu2
            X3_prev = true_future[:, t-1, 2] if t>0 else ctx_history[:, -1, 2]
            mu3 = (X2_prev - 3)**2 + 0.5 * X3_prev
            noise[:, t, 2] = true_future[:, t, 2] - mu3
    
    elif Type == "cyclic_linear_high_dimension":
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

    elif Type == "balanced_tail_linear":
        beta_diag = [0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.7, 0.8]
        beta_par  = {
            (1, 0): 0.3, (2, 0): 0.3,
            (3, 1): 0.3, (4, 1): 0.3,
            (5, 2): 0.3, (6, 2): 0.3,
            (7, 6): 0.3,
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
    
    elif Type == "balanced_tail_linear_beta_2" or Type == "balanced_tail_linear_complicated":
        beta_diag = [0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.2, 0.2]
        if Type == "balanced_tail_linear_beta_2":
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
    
    elif Type == "balanced_tail_non_additive_beta_2":
        beta_diag = [0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.2, 0.2]
        beta_par = {
            (1, 0): 0.3,  # X2 <- X1
            (2, 0): 0.3,  # X3 <- X1
            (3, 1): 0.3,  # X4 <- X2
            (4, 1): 0.3,  # X5 <- X2
            (5, 2): 0.5,  # X6 <- X3
            (6, 2): 0.5,  # X7 <- X3
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
                noise[:, t, j] = (true_future[:, t, j] - infl)/ar  - 0.5
        
    elif Type == "balanced_tail_nonlinear":
        beta_diag = [0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.2, 0.2]
        beta_par = {
            (1, 0): 0.3,  # X2 <- X1
            (2, 0): 0.3,  # X3 <- X1
            (3, 1): 0.3,  # X4 <- X2
            (4, 1): 0.3,  # X5 <- X2
            (5, 2): 0.5,  # X6 <- X3
            (6, 2): 0.5,  # X7 <- X3
            (7, 6): 0.5,  # X8 <- X7
        }
        parents = {
            0: [], 1: [0], 2: [0],
            3: [1], 4: [1],
            5: [2], 6: [2],
            7: [6],
        }
        nonlin_type = {
            1: None,
            2: 'exp',   # X3 uses exp
            3: 'tanh',  # X4 uses tanh
            4: None,
            5: 'exp',
            6: None,
            7: 'tanh',
        }
        for t in range(H):
            for j in range(1, D):
                prev_val = true_future[:, t-1, j] if t>0 else ctx_history[:, -1, j]
                ar = beta_diag[j] * prev_val
                lin_infl = sum(
                    beta_par[(j, k)] *
                    (true_future[:, t-1, k] if t>0 else ctx_history[:, -1, k])
                    for k in parents[j]
                )
                kind = nonlin_type[j]
                if kind == 'exp':
                    infl = torch.exp(lin_infl / 5.0) - 1.0
                elif kind == 'tanh':
                    infl = torch.tanh(lin_infl / 2.0) * lin_infl
                else:
                    infl = lin_infl
                noise[:, t, j] = true_future[:, t, j] - (ar + infl)

    elif Type == "diamond":
        beta_diag = [0.6, 0.5, 0.55, 0.45, 0.5, 
             0.4, 0.35, 0.3,  0.25, 0.2]
        beta_par = {
            # Level0→1
            (1, 0): 0.30,   # X2 ← X1
            (2, 0): 0.35,   # X3 ← X1

            # Level1→2
            (3, 1): 0.25,   # X4 ← X2
            (5, 1): 0.40,   # X6 ← X2
            (4, 2): 0.30,   # X5 ← X3
            (6, 2): 0.45,   # X7 ← X3

            # Level2→3
            (7, 3): 0.50,   # X8 ← X4
            (7, 5): 0.20,   # X8 ← X6
            (8, 4): 0.40,   # X9 ← X5
            (8, 6): 0.30,   # X9 ← X7

            # Level3→4
            (9, 7): 0.60,   # X10 ← X8
            (9, 8): 0.20,   # X10 ← X9
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
    
    elif Type == "diamond_2":
        beta_diag = [0.6, 0.5, 0.55, 0.45, 0.5, 
             0.4, 0.35, 0.3,  0.25, 0.2]
        beta_par = {
            # Level0→1
            (1, 0): 0.30,   # X2 ← X1
            (2, 0): 0.30,   # X3 ← X1

            # Level1→2
            (3, 1): 0.30,   # X4 ← X2
            (5, 1): 0.30,   # X6 ← X2
            (4, 2): 0.30,   # X5 ← X3
            (6, 2): 0.30,   # X7 ← X3

            # Level2→3
            (7, 3): 0.40,   # X8 ← X4
            (7, 5): 0.40,   # X8 ← X6
            (8, 4): 0.40,   # X9 ← X5
            (8, 6): 0.40,   # X9 ← X7

            # Level3→4
            (9, 7): 0.50,   # X10 ← X8
            (9, 8): 0.50,   # X10 ← X9
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
    
    elif Type == "diamond_3_non_additive":
        beta_diag = [0.5] * 10 
        beta_par = {
            (1, 0): 1.0,  # X2 <- X1
            (2, 0): 1.0,  # X3 <- X1
            (3, 1): 0.4,  # X4 <- X2
            (5, 1): 0.4,  # X5 <- X2
            (4, 2): 0.4,  # X6 <- X3
            (6, 2): 0.4,  # X7 <- X3
            (7, 3): 0.4,  # X8 <- X4
            (7, 5): 0.4,  # X8 <- X6
            (8, 4): 0.4,  # X9 <- X5
            (8, 6): 0.4,  # X9 <- X7
            (9, 7): 0.4,  # X10 <- X8
            (9, 8): 0.4,  # X10 <- X9
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
                noise[:, t, j] = (1/(true_future[:, t, j] - infl) - 2)/(5 * torch.exp(ar/3))
    
    elif Type == "diamond_2_square":
        beta_diag = [0.6, 0.5, 0.55, 0.45, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2]
        beta_par = {
            (1,0):0.30, (2,0):0.30,
            (3,1):0.30, (5,1):0.30, (4,2):0.30, (6,2):0.30,
            (7,3):0.40,(7,5):0.40,(8,4):0.40,(8,6):0.40,
            (9,7):0.50,(9,8):0.50
        }
        def f_square(x, gamma=0.10):
            return x + gamma*(x**2)
        for t in range(H):
            for j in range(1, D):
                prev_val = true_future[:, t-1, j] if t>0 else ctx_history[:, -1, j]
                ar = beta_diag[j] * prev_val
                infl = sum(
                    beta_par[(j, k)] *
                    (true_future[:, t-1, k] if t>0 else ctx_history[:, -1, k])
                    for k in parents[j]
                )
                lin = ar + infl
                noise[:, t, j] = true_future[:, t, j] - f_square(lin)
    
    elif Type == "two_layer_feed_forward":
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
                beta_par[(out,hid)] = 0.2 + (hid-3)/10

        for t in range(H):
            for i in range(1,3):
                prev_val = true_future[:, t-1, i] if t>0 else ctx_history[:, -1, i]
                S_t = As[i] * np.sin(2*np.pi*t/Ps[i] + phis[i])
                X_intervention[:, t, i] = beta_diag[i]*prev_val + S_t
            for j in range(3, D):
                prev_val = true_future[:, t-1, j] if t>0 else ctx_history[:, -1, j]
                ar = beta_diag[j] * prev_val
                infl = sum(
                    beta_par[(j, k)] *
                    (true_future[:, t-1, k] if t>0 else ctx_history[:, -1, k])
                    for k in parents[j]
                )
                noise[:, t, j] = true_future[:, t, j] - (ar + infl)
    
    elif Type == "two_layer_feed_forward_2":
        phis = np.linspace(0, 2*np.pi, 10)
        As   = 0.5 + np.arange(10)*0.1
        Ps   = 15 + np.arange(10)*2
        beta_diag = [0.5, 0.4, 0.3] + [0.2] * 7
        sigma     = [0.2] * 10
        beta_par = {}

        X_intervention = ctx_history[:, -(7 + prediction_length):-7, inter_set]
        for hid in range(3,7):
            for inp in range(0,3):
                beta_par[(hid,inp)] = 0.3
        for out in range(7,10):
            for hid in range(3,7):
                beta_par[(out,hid)] = 0.5

        for t in range(H):
            for i in range(1,3):
                prev_val = true_future[:, t-1, i] if t>0 else ctx_history[:, -1, i]
                S_t = As[i] * np.sin(2*np.pi*t/Ps[i] + phis[i])
                X_intervention[:, t, i] = beta_diag[i]*prev_val + S_t
            for j in range(3, D):
                prev_val = true_future[:, t-1, j] if t>0 else ctx_history[:, -1, j]
                ar = beta_diag[j] * prev_val
                infl = sum(
                    beta_par[(j, k)] *
                    (true_future[:, t-1, k] if t>0 else ctx_history[:, -1, k])
                    for k in parents[j]
                )
                noise[:, t, j] = true_future[:, t, j] - (ar + infl)
    
    elif Type == "two_layer_feed_forward_non_additive":
        phis = np.linspace(0, 2*np.pi, 10)
        As   = 0.5 + np.arange(10)*0.1
        Ps   = 15 + np.arange(10)*2
        beta_diag = [0.5] * 10
        sigma     = [0.1] * 10
        beta_par = {}
        for hid in range(3,7):
            for inp in range(0,3):
                beta_par[(hid,inp)] = 0.2
        for out in range(7,10):
            for hid in range(3,7):
                beta_par[(out,hid)] = 0.2

        for t in range(H):
            for i in range(1,3):
                prev_val = true_future[:, t-1, i] if t>0 else ctx_history[:, -1, i]
                S_t = As[i] * np.sin(2*np.pi*t/Ps[i] + phis[i])
                X_intervention[:, t, i] = beta_diag[i]*prev_val + S_t
            for j in range(3, D):
                prev_val = true_future[:, t-1, j] if t>0 else ctx_history[:, -1, j]
                ar = beta_diag[j] * prev_val
                infl = sum(
                    beta_par[(j, k)] *
                    (true_future[:, t-1, k] if t>0 else ctx_history[:, -1, k])
                    for k in parents[j]
                )
                noise[:, t, j] = (true_future[:, t, j] - ar)**2 - 0.5 * torch.abs(infl)

    elif Type == "two_layer_feed_forward_non_additive_2":
        phis = np.linspace(0, 2*np.pi, 10)
        As   = 0.5 + np.arange(10)*0.1
        Ps   = 15 + np.arange(10)*2
        beta_diag = [0.5] * 10
        sigma     = [0.1] * 10
        beta_par = {}
        for hid in range(3,7):
            for inp in range(0,3):
                beta_par[(hid,inp)] = 0.2
        for out in range(7,10):
            for hid in range(3,7):
                beta_par[(out,hid)] = 0.2

        for t in range(H):
            for i in range(1,3):
                prev_val = true_future[:, t-1, i] if t>0 else ctx_history[:, -1, i]
                S_t = As[i] * np.sin(2*np.pi*t/Ps[i] + phis[i])
                X_intervention[:, t, i] = beta_diag[i]*prev_val + S_t
            for j in range(3, D):
                prev_val = true_future[:, t-1, j] if t>0 else ctx_history[:, -1, j]
                ar = beta_diag[j] * prev_val
                infl = sum(
                    beta_par[(j, k)] *
                    (true_future[:, t-1, k] if t>0 else ctx_history[:, -1, k])
                    for k in parents[j]
                )
                noise[:, t, j] = (true_future[:, t, j] - ar)**0.5 - 0.5 * torch.abs(infl)
    
    elif Type == "two_layer_feed_forward_non_additive_3" or Type == "two_layer_feed_forward_non_additive_4":
        phis = np.linspace(0, 2*np.pi, 10)
        As   = 0.5 + np.arange(10)*0.1
        Ps   = 15 + np.arange(10)*2
        beta_diag = [0.5] * 10
        sigma     = [0.2] * 10
        beta_par = {}
        for hid in range(3,7):
            for inp in range(0,3):
                beta_par[(hid,inp)] = 0.2
        for out in range(7,10):
            for hid in range(3,7):
                beta_par[(out,hid)] = 0.2

        for t in range(H):
            for i in range(1,3):
                prev_val = true_future[:, t-1, i] if t>0 else ctx_history[:, -1, i]
                S_t = As[i] * np.sin(2*np.pi*t/Ps[i] + phis[i])
                X_intervention[:, t, i] = beta_diag[i]*prev_val + S_t
            for j in range(3, D):
                prev_val = true_future[:, t-1, j] if t>0 else ctx_history[:, -1, j]
                ar = beta_diag[j] * prev_val
                infl = sum(
                    beta_par[(j, k)] *
                    (true_future[:, t-1, k] if t>0 else ctx_history[:, -1, k])
                    for k in parents[j]
                )
                noise[:, t, j] = (true_future[:, t, j] - ar)/torch.exp(infl/5) - infl/2
    
    elif Type == "two_layer_feed_forward_non_additive_5":
        phis = np.linspace(0, 2*np.pi, 10)
        As   = 0.5 + np.arange(10)*0.1
        Ps   = 15 + np.arange(10)*2
        beta_diag = [0.5, 0.4, 0.3] + [0.2] * 7
        sigma     = [0.15] * 10
        beta_par = {}
        for hid in range(3,7):
            for inp in range(0,3):
                beta_par[(hid,inp)] = 0.3
        for out in range(7,10):
            for hid in range(3,7):
                beta_par[(out,hid)] = 0.5

        for t in range(H):
            for i in range(1,3):
                prev_val = true_future[:, t-1, i] if t>0 else ctx_history[:, -1, i]
                S_t = As[i] * np.sin(2*np.pi*t/Ps[i] + phis[i])
                X_intervention[:, t, i] = beta_diag[i]*prev_val + S_t
            for j in range(3, D):
                prev_val = true_future[:, t-1, j] if t>0 else ctx_history[:, -1, j]
                ar = beta_diag[j] * prev_val
                infl = sum(
                    beta_par[(j, k)] *
                    (true_future[:, t-1, k] if t>0 else ctx_history[:, -1, k])
                    for k in parents[j]
                )
                noise[:, t, j] = (true_future[:, t, j] - ar)/torch.exp(infl/5) - infl/2


        
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
            for i in range(1,3):
                prev_val = true_future[:, t-1, i] if t>0 else ctx_history[:, -1, i]
                S_t = As[i] * np.sin(2*np.pi*t/Ps[i] + phis[i])
                X_intervention[:, t, i] = beta_diag[i]*prev_val + S_t
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

    for t in range(1, H + 1):
        if Type == "cyclic":
            cf[:, t-1, 1] = (
                torch.exp(cf[:, t-1, 0]/5)
                + 0.5 * cf[:, t-1, 1]
                + noise[:, t-1, 1]
            )
            cf[:, t-1, 2] = (
                (cf[:, t-1, 1] - 3)**2
                + 0.5 * cf[:, t-1, 2]
                + noise[:, t-1, 2]
            )
        
        elif Type == "cyclic_linear_high_dimension":
            for j in range(1, D):
                ar = beta_self[j] * cf[:, t-1, j]
                infl = beta[j-1]*cf[:, t-1, j-1] + 0.7 * cf[:, t-1, 0]
                cf[:, t-1, j] = ar + infl + noise[:, t-1, j]

        elif Type == "balanced_tail_linear" or Type == "balanced_tail_linear_beta_2" or Type == "balanced_tail_linear_complicated":
            for j in range(1, D):
                ar_term = beta_diag[j] * cf[:, t-1, j]
                infl_term = sum(
                    beta_par[(j, k)] * cf[:, t-1, k]
                    for k in parents[j]
                )
                cf[:, t-1, j] = ar_term + infl_term + noise[:, t-1, j]
        
        elif Type == "balanced_tail_non_additive_beta_2":
            for j in range(1, D):
                ar_term = beta_diag[j] * cf[:, t-1, j]
                infl_term = sum(
                    beta_par[(j, k)] * cf[:, t-1, k]
                    for k in parents[j]
                )
                cf[:, t-1, j] = ar_term * (noise[:, t-1, j] + 0.5) + infl_term
        
        elif Type == "balanced_tail_nonlinear":
            for j in range(1, D):
                ar_term = beta_diag[j] * cf[:, t-1, j]
                lin_infl = sum(
                    beta_par[(j, k)] * cf[:, t-1, k]
                    for k in parents[j]
                )
                kind = nonlin_type[j]
                if kind == 'exp':
                    infl = torch.exp(lin_infl / 5.0) - 1.0
                elif kind == 'tanh':
                    infl = torch.tanh(lin_infl / 2.0) * lin_infl
                else:
                    infl = lin_infl
                cf[:, t-1, j] = ar_term + infl + noise[:, t-1, j]
        
        elif Type == "diamond" or Type == "diamond_2" or Type == "diamond_3":
            for j in range(1, D):
                ar_term = beta_diag[j] * cf[:, t-1, j]
                infl_term = sum(
                    beta_par[(j, k)] * cf[:, t-1, k]
                    for k in parents[j]
                )
                cf[:, t-1, j] = ar_term + infl_term + noise[:, t-1, j]
        
        elif Type == "diamond_3_non_additive":
            for j in range(1, D):
                ar_term = beta_diag[j] * cf[:, t-1, j]
                infl_term = sum(
                    beta_par[(j, k)] * cf[:, t-1, k]
                    for k in parents[j]
                )
                cf[:, t-1, j] = 1/(2 + 5 * noise[:, t-1, j]) * torch.exp(ar_term/3) + infl_term
        
        elif Type == "diamond_2_square":
            for j in range(1, D):
                ar_term = beta_diag[j] * cf[:, t-1, j]
                infl_term = sum(
                    beta_par[(j, k)] * cf[:, t-1, k]
                    for k in parents[j]
                )
                lin = ar_term + infl_term
                cf[:, t-1, j] = f_square(lin) + noise[:, t-1, j]
        
        elif Type == "two_layer_feed_forward":
            for j in range(3, D):
                ar_term = beta_diag[j] * cf[:, t-1, j]
                infl_term = sum(
                    beta_par[(j, k)] * cf[:, t-1, k]
                    for k in parents[j]
                )
                cf[:, t-1, j] = ar_term + infl_term + noise[:, t-1, j]
        
        elif Type == "two_layer_feed_forward_2":
            for j in range(3, D):
                ar_term = beta_diag[j] * cf[:, t-1, j]
                infl_term = sum(
                    beta_par[(j, k)] * cf[:, t-1, k]
                    for k in parents[j]
                )
                cf[:, t-1, j] = ar_term + infl_term + noise[:, t-1, j]
        
        elif Type == "two_layer_feed_forward_non_additive":
            for j in range(3, D):
                ar_term = beta_diag[j] * cf[:, t-1, j]
                infl_term = sum(
                    beta_par[(j, k)] * cf[:, t-1, k]
                    for k in parents[j]
                )
                cf[:, t-1, j] = (0.5 * torch.abs(infl_term) + noise[:, t-1, j])**0.5 + ar_term
        
        elif Type == "two_layer_feed_forward_non_additive_2":
            for j in range(3, D):
                ar_term = beta_diag[j] * cf[:, t-1, j]
                infl_term = sum(
                    beta_par[(j, k)] * cf[:, t-1, k]
                    for k in parents[j]
                )
                cf[:, t-1, j] = (0.5 * torch.abs(infl_term) + noise[:, t-1, j])**2 + ar_term
        
        elif Type == "two_layer_feed_forward_non_additive_3" or Type == "two_layer_feed_forward_non_additive_4":
            for j in range(3, D):
                ar_term = beta_diag[j] * cf[:, t-1, j]
                infl_term = sum(
                    beta_par[(j, k)] * cf[:, t-1, k]
                    for k in parents[j]
                )
                if Type == "two_layer_feed_forward_non_additive_4":
                    cf[:, t-1, j] = (infl_term/2 + noise[:, t-1, j])* torch.exp(infl_term/5) + ar_term
                else:
                    cf[:, t-1, j] = (infl_term/2 + torch.abs(noise[:, t-1, j]))* torch.exp(infl_term/5) + ar_term
        
        elif Type == "two_layer_feed_forward_non_additive_5":
            for j in range(3, D):
                ar_term = beta_diag[j] * cf[:, t-1, j]
                infl_term = sum(
                    beta_par[(j, k)] * cf[:, t-1, k]
                    for k in parents[j]
                )
                cf[:, t-1, j] = (infl_term/2 + noise[:, t-1, j])* torch.exp(infl_term/5) + ar_term
        
        elif Type == "two_layer_feed_forward_nonlinear":
            for j in range(3, D):
                ar_term = beta_diag[j] * cf[:, t-1, j]
                infl_term = sum(
                    beta_par[(j, k)] * cf[:, t-1, k]
                    for k in parents[j]
                )
                cf[:, t-1, j] = ar_term + f_exp(infl_term) + noise[:, t-1, j]

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
