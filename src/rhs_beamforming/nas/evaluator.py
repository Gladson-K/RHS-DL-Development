import time
import torch
from .objectives import combined_score


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def benchmark_latency(model, sample, warmup=2, runs=5):
    model.eval()
    for _ in range(warmup):
        model(sample)
    if sample.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        model(sample)
    if sample.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / runs


def evaluate_model(model, sample, normalized_rate=0.0):
    latency = benchmark_latency(model, sample)
    params = count_parameters(model)
    score = combined_score(normalized_rate, latency, params)
    return {
        "normalized_rate": float(normalized_rate),
        "latency_ms": float(latency),
        "parameters": int(params),
        "score": float(score),
    }
