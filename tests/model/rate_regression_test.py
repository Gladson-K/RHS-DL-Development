import torch


def test_rate_is_finite():

    rate = calculate_sum_rate(
        H=H,
        W=W,
        noise_var=0.01,
    )

    assert torch.isfinite(rate)

    assert rate >= 0