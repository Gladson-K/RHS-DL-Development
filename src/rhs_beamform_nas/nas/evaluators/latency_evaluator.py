"""Latency evaluator for RHS NAS architectures.

Benchmarks model inference latency in milliseconds across CPU / GPU devices
with warmup and multiple measurement iterations.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class LatencyEvaluator:
    """Evaluates inference latency (ms) for candidate architectures."""

    def __init__(
        self,
        num_warmup: int = 10,
        num_runs: int = 50,
        device: str = "cpu",
    ) -> None:
        self.num_warmup = num_warmup
        self.num_runs = num_runs
        self.device = device

    @torch.no_grad()
    def benchmark_latency(
        self,
        model: nn.Module,
        dummy_input: Optional[torch.Tensor] = None,
        dummy_config: Optional[torch.Tensor] = None,
        input_dim: int = 256,
        config_dim: int = 16,
        batch_size: int = 1,
    ) -> Dict[str, float]:
        """Measure mean and std inference latency per sample in milliseconds.

        Parameters
        ----------
        model:
            PyTorch model.
        dummy_input:
            Optional sample input tensor.
        dummy_config:
            Optional sample config tensor.
        """
        model.eval()
        model.to(self.device)

        if dummy_input is None:
            dummy_input = torch.randn(batch_size, input_dim, device=self.device)
        else:
            dummy_input = dummy_input.to(self.device)

        if dummy_config is None:
            dummy_config = torch.randn(batch_size, config_dim, device=self.device)
        else:
            dummy_config = dummy_config.to(self.device)

        # Warmup
        for _ in range(self.num_warmup):
            _ = model(dummy_input, dummy_config)

        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measurement
        timings = []
        for _ in range(self.num_runs):
            start = time.perf_counter()
            _ = model(dummy_input, dummy_config)
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append((end - start) * 1000.0)  # ms

        mean_latency = float(torch.tensor(timings).mean().item())
        std_latency = float(torch.tensor(timings).std().item()) if len(timings) > 1 else 0.0

        return {
            "latency_ms": round(mean_latency, 4),
            "latency_std_ms": round(std_latency, 4),
            "throughput_fps": round(1000.0 / max(mean_latency, 1e-6) * batch_size, 2),
        }
