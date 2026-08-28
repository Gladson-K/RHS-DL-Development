def combined_score(normalized_rate, latency_ms, parameters,
                   rate_weight=1.0, latency_weight=0.01, parameter_weight=0.001):
    """Simple transparent NAS score; keep raw objectives for Pareto analysis."""
    return (
        rate_weight * normalized_rate
        - latency_weight * latency_ms
        - parameter_weight * (parameters / 1e6)
    )
