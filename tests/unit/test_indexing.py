from pyceles.core.indexing import index_vswf, n_modes, n_scalar, unindex_vswf


def test_n_modes():
    assert n_modes(1) == 2 * 1 * 3
    assert n_modes(2) == 2 * 2 * 4


def test_roundtrip_indices():
    lmax = 5
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            for tau in (1, 2):
                idx = index_vswf(l, m, tau, lmax)
                l2, m2, t2 = unindex_vswf(idx, lmax)
                assert (l, m, tau) == (l2, m2, t2)


def test_tau_blocks_contiguous():
    lmax = 4
    Ns = n_scalar(lmax)
    # first coefficient in CELES ordering is l=1,m=-1,tau=1
    assert index_vswf(1, -1, 1, lmax) == 0
    # tau=2 block starts at offset Ns
    assert index_vswf(1, -1, 2, lmax) == Ns


def test_smuthi_style_contiguous_indexing_order():
    """Port of the core contiguous-order check from SMUTHI's index test."""
    lmax = 5
    idcs = []
    for tau in (1, 2):
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                idcs.append(index_vswf(l, m, tau, lmax))
    assert idcs == list(range(len(idcs)))
