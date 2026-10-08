# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:39:52 2024
@author: hex: phased array + holographic BF in base station communication;
Modified for RHS Elements vs. Sum Rate and Computation Time analysis (Theoretical + DL)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import math
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define FCNN Model with custom test and reconstruction methods
class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128, 64], output_dim=25601):
        super(FCNN, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # Input layer
        layers.append(nn.Linear(prev_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        prev_dim = hidden_dims[0]
        
        # Hidden layers
        for dim in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(0.2))
            prev_dim = dim
        
        # Additional dense layer
        layers.append(nn.Linear(prev_dim, prev_dim // 2))
        layers.append(nn.PReLU())
        layers.append(nn.Dropout(0.1))
        prev_dim = prev_dim // 2
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
    def reconstruct_M(self, output, batch_size):
        """Reconstruct complex-valued M matrix from FCNN output"""
        M_real = output[:, :12800].reshape(batch_size, 20, 64, 10)
        M_imag = output[:, 12800:25600].reshape(batch_size, 20, 64, 10)
        M_complex = torch.complex(M_real, M_imag)
        sum_rate = output[:, 25600]
        return M_complex, sum_rate
    
    def test(self, test_loader, device="cpu"):
        """Evaluate model and compute separate metrics for M_real, M_imag, and sum_rate"""
        self.eval()
        self.to(device)
        total_mse_M_real = 0
        total_mae_M_real = 0
        total_mse_M_imag = 0
        total_mae_M_imag = 0
        total_mse_sum_rate = 0
        total_mae_sum_rate = 0
        total_r2_M_real = 0
        total_r2_M_imag = 0
        total_r2_sum_rate = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = self(batch_x)
                
                # Split output and target
                M_real_pred = output[:, :12800]
                M_imag_pred = output[:, 12800:25600]
                sum_rate_pred = output[:, 25600]
                
                M_real_true = batch_y[:, :12800]
                M_imag_true = batch_y[:, 12800:25600]
                sum_rate_true = batch_y[:, 25600]
                
                # Metrics for M_real
                mse_M_real = F.mse_loss(M_real_pred, M_real_true, reduction='sum').item()
                mae_M_real = F.l1_loss(M_real_pred, M_real_true, reduction='sum').item()
                ss_tot_M_real = torch.sum((M_real_true - M_real_true.mean(dim=0))**2)
                ss_res_M_real = torch.sum((M_real_true - M_real_pred)**2)
                r2_M_real = 1 - (ss_res_M_real / (ss_tot_M_real + 1e-9))
                
                total_mse_M_real += mse_M_real
                total_mae_M_real += mae_M_real
                total_r2_M_real += r2_M_real.item()
                
                # Metrics for M_imag
                mse_M_imag = F.mse_loss(M_imag_pred, M_imag_true, reduction='sum').item()
                mae_M_imag = F.l1_loss(M_imag_pred, M_imag_true, reduction='sum').item()
                ss_tot_M_imag = torch.sum((M_imag_true - M_imag_true.mean(dim=0))**2)
                ss_res_M_imag = torch.sum((M_imag_true - M_imag_pred)**2)
                r2_M_imag = 1 - (ss_res_M_imag / (ss_tot_M_imag + 1e-9))
                
                total_mse_M_imag += mse_M_imag
                total_mae_M_imag += mae_M_imag
                total_r2_M_imag += r2_M_imag.item()
                
                # Metrics for sum_rate
                mse_sum_rate = F.mse_loss(sum_rate_pred, sum_rate_true, reduction='sum').item()
                mae_sum_rate = F.l1_loss(sum_rate_pred, sum_rate_true, reduction='sum').item()
                ss_tot_sum_rate = torch.sum((sum_rate_true - sum_rate_true.mean())**2)
                ss_res_sum_rate = torch.sum((sum_rate_true - sum_rate_pred)**2)
                r2_sum_rate = 1 - (ss_res_sum_rate / (ss_tot_sum_rate + 1e-9))
                
                total_mse_sum_rate += mse_sum_rate
                total_mae_sum_rate += mae_sum_rate
                total_r2_sum_rate += r2_sum_rate.item()
        
        n_samples = len(test_loader.dataset)
        n_elements_M = 12800
        
        avg_mse_M_real = total_mse_M_real / (n_samples * n_elements_M)
        avg_mae_M_real = total_mae_M_real / (n_samples * n_elements_M)
        avg_r2_M_real = total_r2_M_real / len(test_loader)
        
        avg_mse_M_imag = total_mse_M_imag / (n_samples * n_elements_M)
        avg_mae_M_imag = total_mae_M_imag / (n_samples * n_elements_M)
        avg_r2_M_imag = total_r2_M_imag / len(test_loader)
        
        avg_mse_sum_rate = total_mse_sum_rate / n_samples
        avg_mae_sum_rate = total_mae_sum_rate / n_samples
        avg_r2_sum_rate = total_r2_sum_rate / len(test_loader)
        
        return {
            'M_real': {'mse': avg_mse_M_real, 'mae': avg_mae_M_real, 'r2': avg_r2_M_real},
            'M_imag': {'mse': avg_mse_M_imag, 'mae': avg_mae_M_imag, 'r2': avg_r2_M_imag},
            'sum_rate': {'mse': avg_mse_sum_rate, 'mae': avg_mae_sum_rate, 'r2': avg_r2_sum_rate}
        }

# ---- Load DL Model and Scaler ----
model_filename = "best_fcnn_adam_lr0.001.pkl"
scaler_filename = "scaler.pkl"
try:
    fcnn_model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    print(f"Loaded model: {model_filename}")
    print(f"Loaded scaler: {scaler_filename}")
    fcnn_model.eval()
    fcnn_model.to(device)
except FileNotFoundError:
    print("Error: Model or scaler not found. Please ensure the files exist.")
    exit(1)

# Fixed MIMO configuration
Nt = 64
Nr = 64

# ---- Rayleigh Fading Channel Model ----
def RayleighChannel(Nr, Nt, L, train_size):
    H = (torch.randn(train_size, Nr, Nt, L, dtype=torch.cfloat, device=device) +
         1j * torch.randn(train_size, Nr, Nt, L, dtype=torch.cfloat, device=device)) / np.sqrt(2)
    print("Rayleigh Shape:", H.shape)
    return H

# ---- MMSE Beamforming ----
def MMSE_Beamforming(H, noise_var):
    batch_size, Nr_in, Nt_in = H.shape[:3]
    H_H = H.conj().transpose(1, 2)  # [batch_size, Nt, Nr]
    A = torch.bmm(H_H, H) + noise_var * torch.eye(Nt, dtype=torch.cfloat, device=device).unsqueeze(0)
    W = torch.linalg.solve(A, H_H)  # [batch_size, Nt, Nr]
    print("W's shape:", W.shape)
    return W

# ---- Excitation Matrix ----
def genE(lamd, de, Nt, RFno):
    x = torch.arange(Nt, device=device) * de
    y = Nt * de / 2 * torch.ones(Nt, device=device)
    xf = torch.arange(RFno, device=device) * de + de / 2
    yf = Nt * de / 2 * torch.ones(RFno, device=device)
    
    dist = torch.sqrt((x.unsqueeze(1) - xf.unsqueeze(0))**2 + 
                      (y.unsqueeze(1) - yf.unsqueeze(0))**2)
    E = torch.exp(-1j * 2 * np.pi * math.sqrt(3) / lamd * dist)
    print("E's shape:", E.shape)
    return E.unsqueeze(0)

# ---- Gradient Projection Algorithm (GP) ----
def sum_capacity(w, H, M, batch_size):
    Nr_in, L = w.shape[1], w.shape[2]
    obj = torch.zeros(batch_size, dtype=torch.float, device=device)
    
    for l in range(L):
        wt = w[:, :, l]
        H_l = H[:, :, :, l]
        signal = torch.bmm(wt.unsqueeze(1).conj(), torch.bmm(H_l, M))
        power = torch.abs(signal.squeeze(1))**2 / (Nr_in + 1e-9)
        obj += torch.log2(1 + power.sum(dim=1))
    
    max_rate = 80
    scaling_factor = max_rate / (obj.mean() + 1e-9)
    scaling_factor = torch.minimum(scaling_factor, torch.tensor(1.0, device=device))
    return obj * scaling_factor

def GP(H, E, L, hypm, Pt):
    Tno = H.shape[0]
    RFno = E.shape[2]
    
    w = torch.randn(Tno, Nr, L, dtype=torch.cfloat, device=device)
    w = w / (torch.abs(w) + 1e-9)
    m = torch.rand(Tno, Nt, RFno, dtype=torch.float, device=device)
    M = m * E
    print("Initial M shape:", M.shape)
    
    obj0 = torch.tensor(0.0, device=device)
    max_iter = 100
    iter_count = 0
    
    while iter_count < max_iter:
        gradm = torch.zeros(Tno, Nt, RFno, dtype=torch.float, device=device)
        
        for l in range(L):
            wt = w[:, :, l]
            H_l = H[:, :, :, l]
            signal = torch.bmm(wt.unsqueeze(1).conj(), torch.bmm(H_l, M))
            denominator = (torch.abs(signal)**2 + Nr).squeeze(1)
            H_permuted = H_l.permute(0, 2, 1)
            numerator = torch.bmm(H_permuted, wt.unsqueeze(2).conj())
            gradm += (numerator * E / denominator.unsqueeze(1)).real
        
        m = torch.clamp(m + hypm * gradm, 0, 1)
        M = m * E
        
        w_new = torch.zeros(Tno, Nr, L, dtype=torch.cfloat, device=device)
        for l in range(L):
            H_l = H[:, :, :, l]
            w_new[:, :, l] = torch.bmm(H_l, M).mean(dim=2)
        w = w_new / (torch.abs(w_new) + 1e-9)
        
        obj1 = sum_capacity(w, H, M, Tno)
        print("obj1 (sum capacity):", obj1.mean().item())
        if torch.abs(obj1.mean() - obj0) / (obj1.mean() + 1e-9) <= 1e-4:
            print("Convergence reached. Stopping optimization.")
            break
        obj0 = obj1.mean()
        iter_count += 1
    
    return obj1, w, M

# ---- Simulation Parameters ----
dmode = 1  # Change RHS size
L = 3
lamd = 0.0428571  # Wavelength at 30GHz
scale = 1e9
SNR = 23.9
Pt = 10 ** (SNR / 10) * (1 / scale) ** 2
deH = lamd / 5
test_size = 25  # Align with model_creation.py
stepSize1 = 0.5e1
noise_var = 0.001
Nt = 64
Nr = 64

if dmode == 1:
    lopPara = [0, 10, 20, 30, 40]  # Nx parameter for RHS elements
elif dmode == 2:
    lopPara = [2, 4, 6, 8, 10]  # Transmit power

# Storage for results
RateAll_theory = torch.zeros([len(lopPara)], device=device)
TimeAll_theory = np.zeros((len(lopPara)))
RateAll_dl = torch.zeros([len(lopPara)], device=device)
TimeAll_dl = np.zeros((len(lopPara)))
M_means_theory = []
M_means_dl = []

# ---- Generate Input Features for DL Model ----
def generate_features(RFno):
    # Match model input: [L, lamd, scale, SNR, Pt, deH, test_size, RFno, stepSize1, noise_var]
    features = np.array([[L, lamd, scale, SNR, Pt, deH, test_size, RFno, stepSize1, noise_var]])
    return features

# ---- Main Simulation Loop ----
with torch.no_grad():
    for lop in range(len(lopPara)):
        print(f"\nIteration {lop + 1}/{len(lopPara)}: Nx = {lopPara[lop]}")
        Nx = int(lopPara[lop])

        # Adjust RFno to scale with Nx
        RFno = max(1, Nx // 10)  # Increase RF chains with Nx

        # --- Theoretical Simulation ---
        # Generate RHS → User Channel
        H_RHS = RayleighChannel(Nr, Nt, L, test_size) * scale
        
        # MMSE Beamforming
        H_BS = H_RHS.mean(dim=3)  # [test_size, Nr, Nt]
        W_MMSE = MMSE_Beamforming(H_BS, noise_var)
        
        # Compute Effective Channel H_testD
        H_testD = torch.zeros(test_size, Nr, Nt, L, dtype=torch.cfloat, device=device)
        for l in range(L):
            H_testD[:, :, :, l] = torch.bmm(H_RHS[:, :, :, l], W_MMSE)
        
        # Generate Excitation Matrix
        E = genE(lamd, deH, Nt, RFno).repeat(test_size, 1, 1)

        print("H_testD shape:", H_testD.shape)
        print("E shape:", E.shape)
        
        # Run GP
        time_start_theory = time.time()
        gp_rates, w, M_theory = GP(H_testD, E, L, stepSize1, Pt)
        time_end_theory = time.time()

        # Compute mean absolute value of theoretical M
        M_mean_theory = torch.abs(M_theory).mean().item()
        M_means_theory.append(M_mean_theory)

        # Store Theoretical Results
        Rate_theory = gp_rates.mean().item()
        RateAll_theory[lop] = Rate_theory
        TimeAll_theory[lop] = time_end_theory - time_start_theory

        # --- DL Model Prediction ---
        Rate_pred = float('nan')
        M_mean_pred = float('nan')
        M_pred = None
        time_dl = 0.0
        try:
            # Generate features with RFno
            X = generate_features(RFno)
            X_scaled = scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)

            # Predict
            time_start_dl = time.time()
            pred = fcnn_model(X_tensor)  # Shape: (1, 25601)
            M_pred, Rate_pred = fcnn_model.reconstruct_M(pred, batch_size=1)
            time_end_dl = time.time()

            # Compute mean absolute value of predicted M
            M_mean_pred = torch.abs(M_pred).mean().item()
            Rate_pred = Rate_pred.item()
            RateAll_dl[lop] = Rate_pred
            TimeAll_dl[lop] = time_end_dl - time_start_dl
            M_means_dl.append(M_mean_pred)
        except Exception as e:
            print(f"DL prediction failed: {e}")
            Rate_pred = float('nan')
            M_mean_pred = float('nan')
            M_means_dl.append(float('nan'))

        # Print Results
        print(f"RFno: {RFno}")
        print(f"Theoretical M (first sample, shape: {M_theory.shape}):")
        print(M_theory[0].cpu().numpy())  # First sample
        print(f"Theoretical M mean absolute value: {M_mean_theory:.6f}")
        if M_pred is not None:
            print(f"Predicted M (first sample, shape: {M_pred.shape}):")
            print(M_pred[0].cpu().numpy())  # First sample
            print(f"Predicted M mean absolute value: {M_mean_pred:.6f}")
        else:
            print("Predicted M: Not available due to prediction failure")
        print(f"Rate (Theoretical): {Rate_theory:.6f}")
        print(f"Rate (DL): {Rate_pred:.6f}")
        print(f"Computation Time (Theoretical): {TimeAll_theory[lop]:.6f} seconds")
        print(f"Computation Time (DL): {TimeAll_dl[lop]:.6f} seconds")
        print(f"Sum Rate (Theoretical, used for plot): {Rate_theory:.6f}")

# ---- Plot Results ----
# Sum Rate Plot
plt.figure()
plt.plot(lopPara, RateAll_theory.cpu().numpy(), 'b-*', label='RHS (Theoretical)')
plt.plot(lopPara, RateAll_dl.cpu().numpy(), 'r-*', label='RHS (DL Predicted)')
plt.xlabel('Nx: Number of RHS Elements')
plt.ylabel('Sum Rate (bps/Hz)')
plt.title('RHS Elements vs. Sum Rate')
plt.legend()
plt.grid()
plt.savefig('sum_rate_plot.png')
plt.show()

# Computation Time Plot
plt.figure()
plt.semilogy(lopPara, TimeAll_theory / test_size, 'b-*', label='RHS (Theoretical)')
plt.semilogy(lopPara, TimeAll_dl / test_size, 'r-*', label='RHS (DL Inference)')
plt.xlabel('Nx: Number of RHS Elements')
plt.ylabel('Computation Time per Sample (s)')
plt.title('RHS Elements vs. Computation Time')
plt.legend()
plt.grid()
plt.savefig('computation_time_plot.png')
plt.show()