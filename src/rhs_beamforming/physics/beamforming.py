import torch


def normalize_beamformer(w, power=1.0, dim=-1, eps=1e-12):
    """Normalize complex beamforming vectors to a requested power."""
    p = (w.abs() ** 2).sum(dim=dim, keepdim=True)
    scale = torch.sqrt(torch.as_tensor(power, dtype=p.dtype, device=w.device) / p.clamp_min(eps))
    return w * scale


def power_constraint(w, power=1.0, dim=-1, atol=1e-5):
    actual = (w.abs() ** 2).sum(dim=dim)
    target = torch.as_tensor(power, dtype=actual.dtype, device=actual.device)
    return torch.allclose(actual, target.expand_as(actual), atol=atol, rtol=atol)
