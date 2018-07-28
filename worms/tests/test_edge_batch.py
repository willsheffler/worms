import numpy as np
from worms.edge_batch import compute_splices


def test_percomp_splices_1(bbdb_fullsize_prots, spdb):
    names = bbdb_fullsize_prots.query_names('all')
    pairs = [(a, b) for a in names for b in names]
    splices = compute_splices(
        bbdb_fullsize_prots,
        pairs,
        splice_clash_d2=16.0,
        splice_contact_d2=64.0,
        splice_rms_range=6,
        splice_clash_contact_range=60,
        splice_max_rms=0.7,
        splice_ncontact_cut=30,
        skip_on_fail=True
    )
    k0 = ('worms/data/fullsize1.pdb', 'worms/data/fullsize1.pdb')
    k1 = ('worms/data/fullsize1.pdb', 'worms/data/fullsize2.pdb')
    k2 = ('worms/data/fullsize2.pdb', 'worms/data/fullsize1.pdb')
    k3 = ('worms/data/fullsize2.pdb', 'worms/data/fullsize2.pdb')

    # print(repr(list(splices[k2][0])))
    # print(repr(list(splices[k2][1])))
    # print(repr(list(splices[k3][0])))
    # print(repr(list(splices[k3][1])))

    assert np.all(splices[k0][0] == [])
    assert np.all(splices[k0][1] == [])
    assert np.all(splices[k1][0] == [])
    assert np.all(splices[k1][1] == [])
    assert np.all(
        splices[k2][0] == [
            15, 15, 28, 29, 29, 29, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32,
            33, 33, 33, 33, 33, 34, 34, 34
        ]
    )
    assert np.all(
        splices[k2][1] == [
            332, 333, 347, 30, 333, 348, 13, 14, 32, 331, 346, 347, 32, 329,
            332, 347, 348, 12, 330, 333, 346, 348, 13, 331, 347
        ]
    )
    assert np.all(
        splices[k3][0] == [15, 28, 28, 29, 29, 30, 31, 31, 31, 31, 32]
    )
    assert np.all(
        splices[k3][1] ==
        [539, 537, 560, 538, 554, 537, 537, 538, 553, 559, 538]
    )
