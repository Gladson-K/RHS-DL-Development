import torch


def calculate_sum_rate(H, W, noise_var=1e-2, reduction="mean"):
    """
    Generic MU-MIMO sum-rate utility.

    H: [B, K, Nt] complex channel vectors.
    W: [B, Nt, K] complex beamforming matrix.
    """
    if H.ndim != 3 or W.ndim != 3:
        raise ValueError("H must be [B,K,Nt] and W must be [B,Nt,K].")
    if H.shape[0] != W.shape[0] or H.shape[2] != W.shape[1] or H.shape[1] != W.shape[2]:
        raise ValueError("H/W dimensions are incompatible.")

    effective = torch.matmul(H, W)  # [B,K,K]
    signal = effective.diagonal(dim1=1, dim2=2).abs().square()
    total = effective.abs().square().sum(dim=-1)
    interference = (total - signal).clamp_min(0.0)

    noise = torch.as_tensor(noise_var, dtype=signal.dtype, device=signal.device)
    sinr = signal / (interference + noise)
    rates = torch.log2(1.0 + sinr)

    if reduction == "none":
        return rates
    if reduction == "sum":
        return rates.sum(dim=-1)
    return rates.sum(dim=-1).mean()
