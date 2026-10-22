def validate_dimensions(
    H_BS,
    H_RHS,
    H_testD,
    E,
    *,
    Nr: int,
    Nx: int,
    L: int,
    RFno: int,
):
    Nt = Nx ** 2

    assert Nr == 2
    assert Nt == Nx ** 2

    assert H_BS.ndim == 3
    assert H_BS.shape[-2:] == (Nt, Nt)

    assert H_RHS.ndim == 4
    assert H_RHS.shape[1] == Nr
    assert H_RHS.shape[2] == Nt
    assert H_RHS.shape[3] == L

    assert H_testD.ndim == 4
    assert H_testD.shape[1] == L
    assert H_testD.shape[2] == Nr
    assert H_testD.shape[3] == Nt

    assert E.ndim == 3
    assert E.shape[1] == Nt
    assert E.shape[2] == RFno

    return True