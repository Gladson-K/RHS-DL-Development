def test_single_sample_pipeline():

    config = load_config(
        "configs/dataset/debug_20.yaml"
    )

    sample = generate_sample(
        config=config,
        seed=42,
    )

    assert sample["inputs"] is not None
    assert sample["targets"]["M"] is not None
    assert sample["targets"]["w"] is not None
    assert sample["metrics"]["rate"] >= 0