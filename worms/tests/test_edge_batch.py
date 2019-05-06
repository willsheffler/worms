import numpy as np
from worms.edge_batch import compute_splices


def test_percomp_splices_1(bbdb_fullsize_prots, spdb):
    names = bbdb_fullsize_prots.query_names("all")
    pairs = [(a, b) for a in names for b in names]
    splices = compute_splices(
        bbdb_fullsize_prots,
        pairs,
        splice_clash_d2=16.0,
        splice_contact_d2=64.0,
        splice_rms_range=6,
        splice_clash_contact_range=60,
        splice_clash_contact_by_helix=False,
        splice_max_rms=0.7,
        splice_ncontact_cut=30,
        splice_ncontact_no_helix_cut=0,
        splice_nhelix_contacted_cut=0,
        splice_max_chain_length=999999,
        skip_on_fail=True,
        pbar=False,
        verbosity=0,
        parallel=0,
    )
    k0 = ("worms/data/fullsize1.pdb", "worms/data/fullsize1.pdb")
    k1 = ("worms/data/fullsize1.pdb", "worms/data/fullsize2.pdb")
    k2 = ("worms/data/fullsize2.pdb", "worms/data/fullsize1.pdb")
    k3 = ("worms/data/fullsize2.pdb", "worms/data/fullsize2.pdb")

    # print(repr(list(splices[k2][0])))
    # print(repr(list(splices[k2][1])))
    # print(repr(list(splices[k3][0])))
    # print(repr(list(splices[k3][1])))

    assert np.all(splices[k0][0][0] == [])
    assert np.all(splices[k0][0][1] == [])
    assert np.all(splices[k1][0][0] == [])
    assert np.all(splices[k1][0][1] == [])
    assert np.all(
        splices[k2][0][0]
        == [
            537,
            537,
            538,
            538,
            553,
            553,
            553,
            554,
            554,
            554,
            555,
            555,
            556,
            557,
            559,
            559,
            559,
            559,
            560,
            560,
            560,
        ]
    )
    assert np.all(
        splices[k2][0][1]
        == [
            15,
            332,
            330,
            333,
            15,
            347,
            348,
            12,
            330,
            348,
            331,
            346,
            332,
            30,
            32,
            331,
            346,
            347,
            332,
            347,
            348,
        ]
    )
    assert np.all(
        splices[k3][0][0]
        == [
            537,
            537,
            553,
            553,
            554,
            554,
            555,
            555,
            556,
            556,
            556,
            557,
            558,
            559,
            559,
            560,
            560,
            560,
        ]
    )
    assert np.all(
        splices[k3][0][1]
        == [15, 34, 31, 34, 29, 32, 29, 33, 15, 30, 34, 31, 32, 33, 34, 15, 28, 34]
    )
