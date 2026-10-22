from pathlib import Path
import torch


def generate_debug_dataset(samples=20, Nr=2, Nt=256, L=4, RFno=1, seed=42, output=None):
    """Create deterministic tensor data for smoke tests and development."""
    g = torch.Generator().manual_seed(seed)

    H = (torch.randn(samples, L, Nr, Nt, generator=g)
         + 1j * torch.randn(samples, L, Nr, Nt, generator=g)) / 2**0.5

    E = (torch.randn(samples, Nt, RFno, generator=g)
         + 1j * torch.randn(samples, Nt, RFno, generator=g)) / 2**0.5

    targets = {
        "M": E.clone(),
        "w": torch.randn(samples, Nt, RFno, generator=g, dtype=torch.float32)
             .to(torch.cfloat),
    }

    result = {"H": H, "E": E, "targets": targets}

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, path)

    return result
