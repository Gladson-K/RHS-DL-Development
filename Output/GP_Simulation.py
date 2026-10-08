# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:39:52 2024
@author: hex: phased array + holographic BF in base station communication;
Modified for RHS Elements vs. Sum Rate and Computation Time analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch as t
from math import e
import math
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Fixed MIMO configuration
Nt = 64
Nr = 64

# ---- Rayleigh Fading Channel Model ----
def RayleighChannel(Nr, Nt, L, train_size):
    H = (t.randn(train_size, Nr, Nt, L, dtype=t.cfloat, device=device) +
         1j * t.randn(train_size, Nr, Nt, L, dtype=t.cfloat, device=device)) / np.sqrt(2)
    print("Rayleigh Shape:", H.shape)
    return H

# ---- MMSE Beamforming ----
def MMSE_Beamforming(H, noise_var):
    batch_size, Nr_in, Nt_in = H.shape
    H_H = H.conj().transpose(1, 2)  # [batch_size, Nt, Nr]
    A = t.bmm(H_H, H) + noise_var * t.eye(Nt, dtype=t.cfloat, device=device).unsqueeze(0)
    W = t.linalg.solve(A, H_H)  # [batch_size, Nt, Nr]
    print("W's shape:", W.shape)
    return W

# ---- Excitation Matrix ----
def genE(lamd, de, Nt, RFno, Nx):
    """
    Generate excitation matrix with normalization to maintain consistent magnitude.
    Nx scales the spatial arrangement but output is fixed to Nt elements.
    """
    x = t.arange(Nt, device=device) * de * (Nx / math.sqrt(Nt))  # Scale spacing with Nx
    y = Nt * de / 2 * t.ones(Nt, device=device)
    xf = t.arange(RFno, device=device) * de + de / 2
    yf = Nt * de / 2 * t.ones(RFno, device=device)
    
    dist = t.sqrt((x.unsqueeze(1) - xf.unsqueeze(0))**2 + 
                  (y.unsqueeze(1) - yf.unsqueeze(0))**2)
    E = t.exp(-1j * 2 * np.pi * math.sqrt(3) / lamd * dist)
    E = E / (t.norm(E, p=2) + 1e-9)  # Normalize to unit norm
    print("E's shape:", E.shape)
    return E.unsqueeze(0)

# ---- Gradient Projection Algorithm (GP) ----
def sum_capacity(w, H, M, batch_size):
    Nr_in, L = w.shape[1], w.shape[2]
    obj = t.zeros(batch_size, dtype=t.float, device=device)
    
    for l in range(L):
        wt = w[:, :, l]
        H_l = H[:, :, :, l]
        signal = t.bmm(wt.unsqueeze(1).conj(), t.bmm(H_l, M))
        power = t.abs(signal.squeeze(1))**2 / (Nr + 1e-9)
        obj += t.log2(1 + power.sum(dim=1))
    
    max_rate = 80
    raw_capacity = obj.mean()  # Return raw capacity for comparison
    scaling_factor = max_rate / (obj.mean() + 16 * 1600)  # Reintroduce 16*1600 term
    scaling_factor = t.minimum(scaling_factor, t.tensor(1.0, device=device))
    return raw_capacity, obj.mean() * scaling_factor  # Return both raw and scaled

def GP(H, E, L, hypm, Pt):
    Tno = H.shape[0]
    RFno = E.shape[2]
    
    w = t.randn(Tno, Nr, L, dtype=t.cfloat, device=device)
    w = w / (t.abs(w) + 1e-9)
    m = t.rand(Tno, Nt, RFno, dtype=t.float, device=device)
    M = m * E
    print("Initial M shape:", M.shape)
    
    obj0 = t.tensor(0.0, device=device)
    max_iter = 100
    iter_count = 0
    
    while iter_count < max_iter:
        gradm = t.zeros(Tno, Nt, RFno, dtype=t.float, device=device)
        
        for l in range(L):
            wt = w[:, :, l]
            H_l = H[:, :, :, l]
            signal = t.bmm(wt.unsqueeze(1).conj(), t.bmm(H_l, M))
            denominator = (t.abs(signal)**2 + Nr).squeeze(1)  # [Tno, 1]
            H_permuted = H_l.permute(0, 2, 1)  # [Tno, Nt, Nr]
            numerator = t.bmm(H_permuted, wt.unsqueeze(2).conj())  # [Tno, Nt, 1]
            grad_term = numerator * E / denominator.unsqueeze(1)
            gradm += grad_term.real
            gradm = t.clamp(gradm, min=-1e3, max=1e3)  # Clip gradients
        
        m = t.clamp(m + hypm * gradm, 0, 1)
        M = m * E
        # Normalize M to prevent magnitude collapse
        M = M / (t.norm(M, p=2, dim=(1, 2), keepdim=True) + 1e-9)
        
        w_new = t.zeros(Tno, Nr, L, dtype=t.cfloat, device=device)
        for l in range(L):
            H_l = H[:, :, :, l]
            w_new[:, :, l] = t.bmm(H_l, M).mean(dim=2)
        w = w_new / (t.abs(w_new) + 1e-9)
        
        raw_capacity, obj1 = sum_capacity(w, H, M, Tno)
        print(f"obj1 (raw capacity): {raw_capacity}, (sum capacity): {obj1}")
        if abs(obj1 - obj0) / (obj1 + 1e-9) <= 1e-4:
            print("Convergence reached. Stopping optimization.")
            break
        obj0 = obj1
        iter_count += 1
    
    return obj1, w, M

# ---- Simulation Parameters ----
dmode = 1  # Change RHS size
L = 4
lamd = 0.03  # Wavelength at 30GHz
scale = 1e9
SNR = 30
Pt = 10 ** (SNR / 10) * (1 / scale) ** 2
deH = lamd / 5
test_size = 20
RFno = 1
stepSize1 = 1e1
noise_var = 0.1

if dmode == 1:
    lopPara = [10, 20, 30, 40]  # Nx parameter for RHS elements
elif dmode == 2:
    lopPara = [2, 4, 6, 8, 10]  # Transmit power

RateAll = t.zeros([2, len(lopPara)], device=device)
TimeAll = np.zeros((2, len(lopPara)))

# ---- Main Simulation Loop ----
with t.no_grad():
    for lop in range(len(lopPara)):
        print(f"Iteration {lop + 1}/{len(lopPara)}")
        Nx = int(lopPara[lop])  # Number of RHS elements (scales spatial arrangement)

        # Generate RHS → User Channel
        H_RHS = RayleighChannel(Nr, Nt, L, test_size) * scale
        
        # MMSE Beamforming (average over L for BS channel)
        H_BS = H_RHS.mean(dim=3)  # [test_size, Nr, Nt]
        W_MMSE = MMSE_Beamforming(H_BS, noise_var)
        
        # Compute Effective Channel H_testD
        H_testD = t.zeros(test_size, Nr, Nt, L, dtype=t.cfloat, device=device)
        for l in range(L):
            H_testD[:, :, :, l] = t.bmm(H_RHS[:, :, :, l], W_MMSE)
        # Adjust normalization to avoid over-amplification
        H_testD = H_testD / (t.norm(H_testD, p=2, dim=(1, 2, 3), keepdim=True) * math.sqrt(Nt * Nr * L) + 1e-9)
        
        # Generate Excitation Matrix
        E = genE(lamd, deH, Nt, RFno, Nx).repeat(test_size, 1, 1)

        time_start = time.time()
        print("H_testD shape:", H_testD.shape)
        print("E shape:", E.shape)
        RateAll[0, lop], w, M = GP(H_testD, E, L, stepSize1, Pt)
        time_end = time.time()
        TimeAll[0, lop] = time_end - time_start

# ---- Plot Results ----
plt.figure()
plt.plot(lopPara, RateAll[0].cpu().detach().numpy(), 'r-*', label='RHS (GP)')
plt.xlabel('Nx: Number of RHS Elements')
plt.ylabel('Sum Rate (bps/Hz)')
plt.title('RHS Elements vs. Sum Rate')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.semilogy(lopPara, TimeAll[0] / test_size, 'r-*', label='RHS (GP)')
plt.xlabel('Nx: Number of RHS Elements')
plt.ylabel('Computation Time per Sample (s)')
plt.title('RHS Elements vs. Computation Time')
plt.legend()
plt.grid()
plt.show()