"""Eigenvector / EVD (a.k.a. "JEV") beamforming reference.

Adds an *eigen-decomposition* beamforming baseline on top of the GP reference
and the learned NAS models. For every user position we take the effective
channel seen after MMSE combining, decompose its Gram / SVD, and use the
dominant right-singular vectors as the ideal transmit directions. Those ideal
directions are then projected onto the physical RHS excitation basis ``E``
(unit-modulus columns) to obtain a *feasible* holographic beamforming matrix
``M_eig``. The achievable rate is computed with the same surrogate used by the
Gradient-Projection reference so the three references are directly comparable:

    GP (dataset)   vs   EVD / eigen (this module)   vs   NAS model

The physical RHS excitation model is ``M = a * E`` with ``a in [0, 1]`` and
``|E| = 1``, therefore the projection of a desired transmit vector ``f`` onto
RF column ``k`` is ``a_k = clip(Re(conj(E[:,k]) * f), 0, 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except Exception:  # noqa: BLE001  # pragma: no cover
    torch = None


NR = 64
NT = 64
MAX_L = 8
MAX_RFNO = 10
MAX_TEST_SIZE = 20


def _as_torch(x, dtype=torch.complex64):
    if torch is None:
        raise RuntimeError("torch is required for EVD beamforming")
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.as_tensor(np.asarray(x)).to(dtype)


@dataclass
class EVDResult:
    M: torch.Tensor          # [T, Nt, RFno] complex64
    rate: torch.Tensor       # [T] float32
    singular_values: torch.Tensor  # [T, min(Nr,Nt)] float32


class EVDBeamformer:
    """Eigenvector-based holographic beamformer + rate.

    Parameters
    ----------
    rfno : number of RF chains (columns of ``M``/``E``).
    noise_var : noise variance used in the rate surrogate.
    normalize : if True, scale ``M`` so the per-position energy is bounded.
    """

    def __init__(self, rfno: int = MAX_RFNO, noise_var: float = 1e-2,
                 normalize: bool = True, device: str = "cpu"):
        if torch is None:
            raise RuntimeError("torch is required for EVDBeamformer")
        self.rfno = rfno
        self.noise_var = noise_var
        self.normalize = normalize
        self.device = device

    # -- core ------------------------------------------------------------- #
    def beamform(self, H_eff: torch.Tensor, E: torch.Tensor) -> EVDResult:
        """Eigen-beamform a batch of effective channels.

        H_eff : [T, Nr, Nt] complex, effective channel (MMSE-combined).
        E     : [T, Nt, RFno] complex, unit-modulus RHS excitation.
        """
        H_eff = _as_torch(H_eff).to(self.device)
        E = _as_torch(E).to(self.device)
        T, Nr, Nt = H_eff.shape
        R = min(Nr, Nt)

        # Right singular vectors V of H_eff = U S V^H  (batched CPU SVD).
        try:
            _, s, Vh = torch.linalg.svd(H_eff, full_matrices=False)
        except Exception:  # noqa: BLE001  # pragma: no cover - fallback
            _, s, Vh = torch.svd(H_eff)
        V = Vh.conj().transpose(-1, -2)  # [T, Nt, R]
        k = min(self.rfno, V.shape[-1])

        F = V[:, :, :k]  # [T, Nt, k] ideal transmit directions
        # Project each stream onto the matching RF excitation column.
        E_k = E[:, :, :k]  # [T, Nt, k]
        amp = (torch.conj(E_k) * F).real.clamp(0.0, 1.0)  # [T, Nt, k]
        M = amp * E_k
        if self.normalize:
            energy = M.abs().pow(2).sum(dim=1, keepdim=True).clamp_min(1e-9)  # [T,1,k]
            M = M / energy.sqrt() * np.sqrt(Nt / max(k, 1))
        return EVDResult(M=M, rate=torch.zeros(T), singular_values=s[:, :R])

    # -- rate ------------------------------------------------------------- #
    def rate(self, H_seq: torch.Tensor, M: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """GP-compatible surrogate rate.

        H_seq : [T, Nr, Nt, L] complex
        M     : [T, Nt, RFno] complex
        w     : [T, Nr, L]   complex (receive combiner, e.g. GP ``w``)
        """
        H_seq = _as_torch(H_seq).to(self.device)
        M = _as_torch(M).to(self.device)
        w = _as_torch(w).to(self.device)
        T, Nr, _Nt, L = H_seq.shape

        obj = torch.zeros(T, dtype=torch.float32, device=self.device)
        for l in range(L):
            wl = w[:, :, l]                      # [T, Nr]
            Hl = H_seq[:, :, :, l]               # [T, Nr, Nt]
            # [T, Nr, Nt] x [T, Nt, rfno] -> [T, Nr, rfno]
            HM = torch.einsum("tnm,tmk->tnk", Hl, M)
            sig = torch.einsum("tn,tnk->tk", wl.conj(), HM)  # [T, rfno]
            power = sig.abs().pow(2) / Nr        # [T, rfno]
            obj = obj + torch.log2(1.0 + power.sum(dim=-1))
        return obj

    # -- high-level ------------------------------------------------------- #
    def reference(self, H_seq: torch.Tensor, E_seq: torch.Tensor,
                  w: torch.Tensor) -> EVDResult:
        """Compute EVD beamformer + rate for a batch of samples.

        H_seq : [B, L, Nr, Nt] complex  (subcarriers for one best position)
        E_seq : [B, Nt, RFno] complex
        w     : [B, Nr, L] complex
        """
        H_seq = _as_torch(H_seq).to(self.device)
        E_seq = _as_torch(E_seq).to(self.device)
        w = _as_torch(w).to(self.device)
        # Average effective channel across subcarriers for the SVD direction.
        H_eff = H_seq.mean(dim=1)  # [B, Nr, Nt]
        res = self.beamform(H_eff, E_seq)
        # Expand M over subcarriers for the rate surrogate: M is shared per sample.
        Hp = H_seq.permute(0, 2, 3, 1).contiguous()  # [B, Nr, Nt, L]
        res.rate = self.rate(Hp, res.M, w)
        return res


def eigenvalue_spectrum(H_eff, top_k: int = 4) -> np.ndarray:
    """Eigenvalues (descending) of the effective-channel Gram ``H^H H``.

    Convenience helper for analysis / reporting. Accepts numpy or torch.
    """
    H = _as_torch(H_eff)
    G = H.conj().transpose(-1, -2) @ H
    evals = torch.linalg.eigvalsh(G).flip(-1)
    return evals[..., :top_k].real.cpu().numpy()
