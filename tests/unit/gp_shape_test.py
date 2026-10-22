def test_gp_tensor_shapes(system_config):

    B = 2

    Nr = system_config["Nr"]
    Nt = system_config["Nt"]
    L = system_config["L"]
    RFno = system_config["RFno"]

    H_testD = torch.randn(
        B,
        L,
        Nr,
        Nt,
        dtype=torch.cfloat,
    )

    E = torch.randn(
        B,
        Nt,
        RFno,
        dtype=torch.cfloat,
    )

    assert H_testD.shape == (
        B,
        L,
        Nr,
        Nt,
    )

    assert E.shape == (
        B,
        Nt,
        RFno,
    )