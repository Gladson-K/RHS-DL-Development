import pytest
import torch


@pytest.fixture
def device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


@pytest.fixture
def system_config():
    return {
        "Nr": 2,
        "Nx": 16,
        "Nt": 256,
        "L": 4,
        "RFno": 1,
    }