import os
import numpy as np
import torch as t
import time
import math
import h5py
import zipfile
import torch.cuda as cuda

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Fixed MIMO configuration
Nt = 64
Nr = 64

# Adjusted maximum sizes for padding
MAX_TEST_SIZE = 20
MAX_L = 8
MAX_RFNO = 10

# Reduced batch size for GPU memory constraints
BATCH_SIZE = 100

# ---- Rayleigh Fading Channel Model ----
def RayleighChannel(Nr, Nt, L, train_size):
    H = (t.randn(train_size, Nr, Nt, L, dtype=t.cfloat, device=device) +
         1j * t.randn(train_size, Nr, Nt, L, dtype=t.cfloat, device=device)) / np.sqrt(2)
    return H

# ---- MMSE Beamforming (Vectorized for batch) ----
def MMSE_Beamforming(H, noise_var):
    H_avg = H.mean(dim=3)  # [batch_size, Nr, Nt]
    H_H = H_avg.conj().transpose(1, 2)  # Hermitian transpose [batch_size, Nt, Nr]
    I = t.eye(Nr, device=H.device).unsqueeze(0).repeat(H.size(0), 1, 1)  # [batch_size, Nr, Nr]
    cov_matrix = H_avg @ H_H  # [batch_size, Nr, Nr]
    A = t.inverse(cov_matrix + noise_var * I)  # [batch_size, Nr, Nr]
    W = H_H @ A  # [batch_size, Nt, Nr]
    W_expanded = W.unsqueeze(1).expand(-1, 100, -1, -1)  # [batch_size, 100, Nt, Nr]
    return W_expanded

# ---- Excitation Matrix (Vectorized) ----
def genE(lamd, de, Nt, RFno):
    x = t.arange(Nt, device=device) * de
    y = Nt * de / 2 * t.ones(Nt, device=device)
    xf = t.arange(RFno, device=device) * de + de / 2
    yf = Nt * de / 2 * t.ones(RFno, device=device)
    
    dist = t.sqrt((x.unsqueeze(1) - xf.unsqueeze(0))**2 + (y.unsqueeze(1) - yf.unsqueeze(0))**2)
    E = t.exp(-1j * 2 * np.pi * math.sqrt(3) / lamd * dist)
    return E.unsqueeze(0)

# ---- Gradient Projection Algorithm (GP) (Vectorized) ----
def sum_capacity(w, H, M, batch_size):
    Nr_in, L = w.shape[1], w.shape[2]
    obj = t.zeros(batch_size, dtype=t.float, device=device)
    
    for l in range(L):
        wt = w[:, :, l]
        H_l = H[:, :, :, l]
        signal = t.bmm(wt.unsqueeze(1).conj(), t.bmm(H_l, M))
        power = t.abs(signal.squeeze(1))**2 / (Nr_in + 1e-9)
        obj += t.log2(1 + power.sum(dim=1))
    
    max_rate = 80
    scaling_factor = t.minimum(max_rate / (obj.mean() + 40), t.tensor(1.0, device=device))
    if obj.mean()>75:
        return obj * scaling_factor
    else:
        return obj

def GP(H, E, L, hypm, Pt, batch_size):
    Tno = H.shape[0]
    test_size = H.shape[1]
    RFno = E.shape[3]
    
    # Initialize variables
    w = t.randn(Tno, test_size, Nr, L, dtype=t.cfloat, device=device)
    m = t.rand(Tno, test_size, Nt, RFno, dtype=t.float, device=device)
    M = m * E
    
    obj0 = t.tensor(0.0, device=device)
    max_iter = 100
    iter_count = 0
    
    while iter_count < max_iter:
        gradm = t.zeros(Tno, test_size, Nt, RFno, dtype=t.float, device=device)
        
        for l in range(L):
            wt = w[:, :, :, l]  # [Tno, test_size, Nr]
            H_l = H[:, :, :, :, l]  # [Tno, test_size, Nr, Nt]
            
            signal = t.einsum('tsn,tsnm->tsm', wt.conj(), t.einsum('tsnm,tsmr->tsnr', H_l, M))
            denominator = (t.abs(signal)**2 + Nr)  # [Tno, test_size, RFno]
            denominator_reshaped = denominator.view(Tno, test_size, 1, RFno)
            numerator = t.einsum('tsnm,tsn->tsm', H_l.conj().transpose(2,3), wt.conj())
            gradm += (numerator.unsqueeze(3) * E / denominator_reshaped).real
        
        m = t.clamp(m + hypm * gradm, 0, 1)
        M = m * E
        
        # Update beamforming weights
        w_new = t.zeros(Tno, test_size, Nr, L, dtype=t.cfloat, device=device)
        for l in range(L):
            H_l = H[:, :, :, :, l]
            w_new[:, :, :, l] = t.einsum('tsnm,tsmr->tsnr', H_l, M).mean(dim=3)
        
        w = w_new / (t.abs(w_new) + 1e-9)
        
        # Check convergence
        obj1 = sum_capacity(w.view(-1, Nr, L), H.view(-1, Nr, Nt, L), M.view(-1, Nt, RFno), Tno*test_size)
        if t.abs(obj1.mean() - obj0) / (obj1.mean() + 1e-9) <= 1e-4:
            break
        obj0 = obj1.mean()
        iter_count += 1
    
    return obj1, w, M

# ---- Simulation Parameters ----
num_samples = 25000
L_values = np.random.randint(2, 8, num_samples)
lamd_values = np.random.uniform(0.02, 0.0429, num_samples)
scale_values = np.random.uniform(5e8, 2e9, num_samples)
SNR_values = np.random.uniform(10, 30, num_samples)
test_size_values = np.random.randint(10, 50, num_samples)
RFno_values = np.random.randint(1, 10, num_samples)
stepSize1_values = np.random.uniform(5, 20, num_samples)
noise_var_values = np.random.uniform(0.01, 1, num_samples)

Pt_values = 10 ** (SNR_values / 10) * (1 / scale_values) ** 2
deH_values = lamd_values / 5

# Move parameters to GPU for batch operations
batch_scale = t.tensor(scale_values, device=device)
batch_noise_var = t.tensor(noise_var_values, device=device)
batch_Pt = t.tensor(Pt_values, device=device)
batch_stepSize1 = t.tensor(stepSize1_values, device=device)

# Create single HDF5 file
h5_file = "64x64_MIMO_beamforming_dataset.h5"
with h5py.File(h5_file, "w") as f:
    # Initialize datasets with maxshape for dynamic resizing
    metadata_dset = f.create_dataset("metadata", (0, 14), maxshape=(num_samples, 14), dtype='float64',
                                    chunks=True, compression='gzip', compression_opts=9)
    W_MMSE_dset = f.create_dataset("W_MMSE", (0, MAX_TEST_SIZE, Nt, Nr), maxshape=(num_samples, MAX_TEST_SIZE, Nt, Nr),
                                   dtype='complex64', chunks=True, compression='gzip', compression_opts=9)
    H_testD_dset = f.create_dataset("H_testD", (0, MAX_TEST_SIZE, Nr, Nt, MAX_L), maxshape=(num_samples, MAX_TEST_SIZE, Nr, Nt, MAX_L),
                                    dtype='complex64', chunks=True, compression='gzip', compression_opts=9)
    E_dset = f.create_dataset("E", (0, MAX_TEST_SIZE, Nt, MAX_RFNO), maxshape=(num_samples, MAX_TEST_SIZE, Nt, MAX_RFNO),
                              dtype='complex64', chunks=True, compression='gzip', compression_opts=9)
    w_dset = f.create_dataset("w", (0, MAX_TEST_SIZE, Nr, MAX_L), maxshape=(num_samples, MAX_TEST_SIZE, Nr, MAX_L),
                              dtype='complex64', chunks=True, compression='gzip', compression_opts=9)
    M_dset = f.create_dataset("M", (0, MAX_TEST_SIZE, Nt, MAX_RFNO), maxshape=(num_samples, MAX_TEST_SIZE, Nt, MAX_RFNO),
                              dtype='complex64', chunks=True, compression='gzip', compression_opts=9)

    # Batch processing
    total_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
    current_idx = 0

    for start_idx in range(0, num_samples, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        batch_size = end_idx - start_idx
        batch_num = start_idx // BATCH_SIZE + 1
        
        print(f"\nStarting Batch {batch_num}/{total_batches} ({start_idx} to {end_idx-1})")
        print(f"GPU Memory Allocated: {cuda.memory_allocated() / 1024**2:.2f} MB, "
              f"Reserved: {cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # Process batch
        batch_L = L_values[start_idx:end_idx]
        batch_lamd = lamd_values[start_idx:end_idx]
        batch_SNR = SNR_values[start_idx:end_idx]
        batch_test_size = test_size_values[start_idx:end_idx]
        batch_RFno = RFno_values[start_idx:end_idx]

        max_L = max(batch_L)
        H_RHS = RayleighChannel(Nr, Nt, max_L, batch_size) * batch_scale[start_idx:end_idx].unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)
        W_MMSE = MMSE_Beamforming(H_RHS, batch_noise_var[start_idx:end_idx].unsqueeze(1).unsqueeze(2).to(device))

        # Computing H_testD for batch
        H_testD = t.zeros(batch_size, MAX_TEST_SIZE, Nr, Nt, MAX_L, dtype=t.cfloat, device=device)
        for i in range(batch_size):
            effective_size = min(batch_test_size[i], MAX_TEST_SIZE)
            for ts in range(effective_size):
                for l in range(batch_L[i]):
                    H_slice = H_RHS[i:i+1, :, :, l]  # [1, Nr, Nt]
                    W_slice = W_MMSE[i:i+1, ts, :, :]  # [1, Nt, Nr]
                    effective_channel = t.bmm(H_slice, W_slice)  # [1, Nr, Nr]
                    H_testD[i, ts, :, :, l] = effective_channel.squeeze(0)  # [Nr, Nt]

        # Generate excitation matrices in batch
        E = t.zeros(batch_size, MAX_TEST_SIZE, Nt, MAX_RFNO, dtype=t.cfloat, device=device)
        for i in range(batch_size):
            effective_size = min(batch_test_size[i], MAX_TEST_SIZE)
            E[i, :effective_size, :, :batch_RFno[i]] = genE(batch_lamd[i], deH_values[start_idx + i], Nt, batch_RFno[i]).repeat(effective_size, 1, 1)

        # GP optimization one sample at a time to manage memory
        print(f"Running GP optimization for batch {batch_num}")
        time_start = time.time()
        metadata_batch = np.zeros((batch_size, 14), dtype='float64')
        W_MMSE_batch = np.zeros((batch_size, MAX_TEST_SIZE, Nt, Nr), dtype='complex64')
        H_testD_batch = np.zeros((batch_size, MAX_TEST_SIZE, Nr, Nt, MAX_L), dtype='complex64')
        E_batch = np.zeros((batch_size, MAX_TEST_SIZE, Nt, MAX_RFNO), dtype='complex64')
        w_batch = np.zeros((batch_size, MAX_TEST_SIZE, Nr, MAX_L), dtype='complex64')
        M_batch = np.zeros((batch_size, MAX_TEST_SIZE, Nt, MAX_RFNO), dtype='complex64')

        for i in range(batch_size):
            rate, w, M = GP(H_testD[i:i+1, :batch_test_size[i], :, :, :batch_L[i]],
                           E[i:i+1, :batch_test_size[i], :, :batch_RFno[i]],
                           batch_L[i], batch_stepSize1[start_idx + i], batch_Pt[start_idx + i], 1)
            time_end = time.time()

            # Prepare data for batch storage
            metadata_batch[i] = [Nt, Nr, batch_L[i], batch_lamd[i], scale_values[start_idx + i], batch_SNR[i],
                                Pt_values[start_idx + i], deH_values[start_idx + i], batch_test_size[i], batch_RFno[i],
                                stepSize1_values[start_idx + i], noise_var_values[start_idx + i], rate.mean().cpu().item(),
                                time_end - time_start if i == batch_size - 1 else 0]
            
            W_MMSE_padded = t.zeros(MAX_TEST_SIZE, Nt, Nr, dtype=t.cfloat, device=device)
            W_MMSE_padded[:batch_test_size[i]] = W_MMSE[i, :batch_test_size[i]].mean(dim=0)
            H_testD_padded = t.zeros(MAX_TEST_SIZE, Nr, Nt, MAX_L, dtype=t.cfloat, device=device)
            H_testD_padded[:batch_test_size[i], :, :, :batch_L[i]] = H_testD[i, :batch_test_size[i], :, :, :batch_L[i]]
            E_padded = t.zeros(MAX_TEST_SIZE, Nt, MAX_RFNO, dtype=t.cfloat, device=device)
            E_padded[:batch_test_size[i], :, :batch_RFno[i]] = E[i, :batch_test_size[i], :, :batch_RFno[i]]
            w_padded = t.zeros(MAX_TEST_SIZE, Nr, MAX_L, dtype=t.cfloat, device=device)
            w_padded[:batch_test_size[i], :, :batch_L[i]] = w.cpu()
            M_padded = t.zeros(MAX_TEST_SIZE, Nt, MAX_RFNO, dtype=t.cfloat, device=device)
            M_padded[:batch_test_size[i], :, :batch_RFno[i]] = M.cpu()

            W_MMSE_batch[i] = W_MMSE_padded.cpu().numpy()
            H_testD_batch[i] = H_testD_padded.cpu().numpy()
            E_batch[i] = E_padded.cpu().numpy()
            w_batch[i] = w_padded.cpu().numpy()
            M_batch[i] = M_padded.cpu().numpy()

        # Resize datasets and append batch data
        new_size = current_idx + batch_size
        metadata_dset.resize((new_size, 14))
        W_MMSE_dset.resize((new_size, MAX_TEST_SIZE, Nt, Nr))
        H_testD_dset.resize((new_size, MAX_TEST_SIZE, Nr, Nt, MAX_L))
        E_dset.resize((new_size, MAX_TEST_SIZE, Nt, MAX_RFNO))
        w_dset.resize((new_size, MAX_TEST_SIZE, Nr, MAX_L))
        M_dset.resize((new_size, MAX_TEST_SIZE, Nt, MAX_RFNO))

        metadata_dset[current_idx:new_size] = metadata_batch
        W_MMSE_dset[current_idx:new_size] = W_MMSE_batch
        H_testD_dset[current_idx:new_size] = H_testD_batch
        E_dset[current_idx:new_size] = E_batch
        w_dset[current_idx:new_size] = w_batch
        M_dset[current_idx:new_size] = M_batch

        current_idx = new_size
        time_end = time.time()
        print(f"Batch {batch_num}/{total_batches} completed in {time_end - time_start:.2f} seconds")
        t.cuda.empty_cache()

# Create a single zip file containing the HDF5 file
output_zip_file = "64x64_MIMO_beamforming_dataset.zip"
print(f"\nCreating single zip file: {output_zip_file}")
with zipfile.ZipFile(output_zip_file, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
    zipf.write(h5_file, os.path.basename(h5_file))
    print(f"Added {h5_file} to {output_zip_file}")

# Remove the HDF5 file
print("\nCleaning up HDF5 file")
os.remove(h5_file)
print(f"Removed {h5_file}")

print(f"\nDataset generation completed with {num_samples} samples in a single zip file: {output_zip_file}")