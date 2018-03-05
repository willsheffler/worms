import pytest
import pickle
import numpy as np
from homog import hrot, htrans, axis_angle_of, axis_ang_cen_of
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from .. import *
from homog.sym import icosahedral_axes as IA
import time
try:
    import pyrosetta
    HAVE_PYROSETTA = True
    try:
        import pyrosetta.distributed
        HAVE_PYROSETTA_DISTRIBUTED = True
    except ImportError:
        HAVE_PYROSETTA_DISTRIBUTED = False
except ImportError:
    HAVE_PYROSETTA = HAVE_PYROSETTA_DISTRIBUTED = False


only_if_pyrosetta = pytest.mark.skipif('not HAVE_PYROSETTA')


@only_if_pyrosetta
def test_sym_bug(c1pose, c2pose):
    helix = Spliceable(
        c1pose, sites=[((1, 2, 3, 4,), 'N'), ((9, 10, 11, 13), 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:3', 'N'), ('1,-1:', 'C'),
                                      ('2,:3', 'N'), ('2,-1:', 'C')])
    segments = [Segment([helix], exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([dimer], entry='N', exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([helix], entry='N'), ]
    wnc = grow(segments, Cyclic(3, lever=200), thresh=1, verbosity=1)
    assert len(wnc) == 6
    print(wnc.scores)
    p = wnc.pose(0, align=1, end=1)
    # vis.showme(p)
    # show_with_axis(wnc, 0)
    # assert 0
    # q = wnc.pose(4)
    # vis.showme(p, name='carterr')
    # vis.showme(q, name='angerr')
    assert residue_sym_err(wnc.pose(0, end=True), 120, 2, 46, 6) < 1.0
    # assert 0


@only_if_pyrosetta
def test_SpliceSite(pose, c3pose):
    assert len(pose) == 7
    ss = SpliceSite(1, 'N')
    spliceable = Spliceable(pose, [])
    spliceablec3 = Spliceable(c3pose, [])
    assert 1 == ss.resid(1, spliceable.body)
    assert pose.size() == ss.resid(-1, spliceable.body)
    assert ss._resids(spliceable) == [1]
    assert SpliceSite('1:7', 'N')._resids(spliceable) == [1, 2, 3, 4, 5, 6, 7]
    assert SpliceSite(':7', 'N')._resids(spliceable) == [1, 2, 3, 4, 5, 6, 7]
    assert SpliceSite('-3:-1', 'N')._resids(spliceable) == [5, 6, 7]
    assert SpliceSite('-3:', 'N')._resids(spliceable) == [5, 6, 7]
    assert SpliceSite(':2', 'N')._resids(spliceable) == [1, 2]
    assert SpliceSite(':-5', 'N')._resids(spliceable) == [1, 2, 3]
    assert SpliceSite('::2', 'N')._resids(spliceable) == [1, 3, 5, 7]
    with pytest.raises(ValueError): SpliceSite(
        '-1:-3', 'N')._resids(spliceable)
    with pytest.raises(ValueError): SpliceSite('-1:3', 'N')._resids(spliceable)
    assert SpliceSite([1, 2, 3], 'N', 1)._resids(spliceablec3) == [1, 2, 3]
    assert SpliceSite([1, 2, 3], 'N', 2)._resids(spliceablec3) == [10, 11, 12]
    assert SpliceSite([1, 2, 3], 'N', 3)._resids(spliceablec3) == [19, 20, 21]


@only_if_pyrosetta
def test_spliceable(c2pose):
    site1 = SpliceSite([1, 2, 3], 'N', 1)
    site2 = SpliceSite([1, 2, 3], 'N', 2)
    dimer = Spliceable(c2pose, sites=[site1, site2])
    assert dimer.sites[0]._resids(dimer) == [1, 2, 3]
    assert dimer.sites[1]._resids(dimer) == [13, 14, 15]

    site1 = {'sele': [1, 2, 3], 'polarity': 'N', 'chain': 1}
    site2 = {'sele': [1, 2, 3], 'polarity': 'N', 'chain': 2}
    dimer = Spliceable(c2pose, sites=[site1, site2])
    assert dimer.sites[0]._resids(dimer) == [1, 2, 3]
    assert dimer.sites[1]._resids(dimer) == [13, 14, 15]

    site1 = ([1, 2, 3], 'N', 1)
    site2 = ([1, 2, 3], 'N', 2)
    dimer = Spliceable(c2pose, sites=[site1, site2])
    assert dimer.sites[0]._resids(dimer) == [1, 2, 3]
    assert dimer.sites[1]._resids(dimer) == [13, 14, 15]

    site1 = (':3', 'N')
    site2 = ('2,:3', 'N')
    dimer = Spliceable(c2pose, sites=[site1, site2])
    assert dimer.sites[0]._resids(dimer) == [1, 2, 3]
    assert dimer.sites[1]._resids(dimer) == [13, 14, 15]


@pytest.mark.skip  # if('not HAVE_PYROSETTA')
def test_spliceable_pickle(tmpdir, c2pose):
    site1 = SpliceSite([1, 2, 3], 'N', 1)
    site2 = SpliceSite([1, 2, 3], 'N', 2)
    dimer = Spliceable(c2pose, sites=[site1, site2])
    pickle.dump(dimer, open(str(os.path.join(tmpdir, 'test.pickle')), 'wb'))
    dimer2 = pickle.load(open(str(os.path.join(tmpdir, 'test.pickle')), 'rb'))
    assert str(dimer) == str(dimer2)


def test_geom_check():
    SX = Cyclic
    I = np.identity(4)
    rotx1rad = hrot([1, 0, 0], 1)
    transx10 = htrans([10, 0, 0])
    randaxes = np.random.randn(1, 3)

    assert 0 == SX('c1').score([I, I])
    assert 0.001 > abs(50 - SX('c1').score([I, rotx1rad]))
    assert 1e-5 > abs(SX('c2').score([I, hrot([1, 0, 0], np.pi)]))

    score = Cyclic('c2').score([I, hrot(randaxes, np.pi)])
    assert np.allclose(0, score, atol=1e-5, rtol=1)

    score = Cyclic('c3').score(
        [I, hrot(randaxes, np.pi * 2 / 3)])
    assert np.allclose(0, score, atol=1e-5, rtol=1)

    score = Cyclic('c4').score([I, hrot(randaxes, np.pi / 2)])
    assert np.allclose(0, score, atol=1e-5, rtol=1)


@only_if_pyrosetta
def test_segment_geom(c1pose):
    "currently only a basic sanity checkb... only checks translation distances"
    body = c1pose
    stubs = util.get_bb_stubs(body)
    assert stubs.shape == (body.size(), 4, 4)

    nsplice = SpliceSite(polarity='N', sele=[1, 2, ])
    csplice = SpliceSite(polarity='C', sele=[9, 10, 11, 12, 13])
    Npairs0 = len(nsplice.selections) * len(csplice.selections)

    # N to N and C to C invalid, can't splice to same
    spliceable = Spliceable(body, sites=[nsplice, csplice])
    with pytest.raises(ValueError):
        seg = Segment([spliceable], entry='N', exit='N')
    with pytest.raises(ValueError):
        seg = Segment([spliceable] * 3, entry='C', exit='C')

    # add some extra splice sites
    Nexsite = 2
    spliceable = Spliceable(body, sites=[nsplice, csplice] * Nexsite)

    # test beginning segment.. only has exit
    seg = Segment([spliceable], exit='C')
    assert seg.x2exit.shape == (Nexsite * len(csplice.selections), 4, 4)
    assert seg.x2orgn.shape == (Nexsite * len(csplice.selections), 4, 4)
    assert np.all(seg.x2exit[..., 3, :3] == 0)
    assert np.all(seg.x2exit[..., 3, 3] == 1)
    for e2x, e2o, ir, jr in zip(seg.x2exit, seg.x2orgn,
                                seg.entryresid, seg.exitresid):
        assert ir == -1
        assert np.allclose(e2o, np.eye(4))
        assert np.allclose(e2x, stubs[jr - 1])

    # test middle segment with entry and exit
    seg = Segment([spliceable], 'N', 'C')
    assert seg.x2exit.shape == (Nexsite**2 * Npairs0, 4, 4)
    assert seg.x2orgn.shape == (Nexsite**2 * Npairs0, 4, 4)
    assert np.all(seg.x2exit[..., 3, :3] == 0)
    assert np.all(seg.x2exit[..., 3, 3] == 1)
    for e2x, e2o, ir, jr in zip(seg.x2exit, seg.x2orgn,
                                seg.entryresid, seg.exitresid):
        assert np.allclose(stubs[ir - 1] @ e2o, np.eye(4), atol=1e-5)
        assert np.allclose(stubs[ir - 1] @ e2x, stubs[jr - 1], atol=1e-5)

    # test ending segment.. only has entry
    seg = Segment([spliceable], entry='N')
    assert seg.x2exit.shape == (Nexsite * len(nsplice.selections), 4, 4)
    assert seg.x2orgn.shape == (Nexsite * len(nsplice.selections), 4, 4)
    assert np.all(seg.x2exit[..., 3, :3] == 0)
    assert np.all(seg.x2exit[..., 3, 3] == 1)
    for e2x, e2o, ir, jr in zip(seg.x2exit, seg.x2orgn,
                                seg.entryresid, seg.exitresid):
        assert jr == -1
        assert np.allclose(e2o, e2x)
        assert np.allclose(e2o @ stubs[ir - 1], np.eye(4), atol=1e-5)

    # test now with multiple spliceables input to segment
    Nexbody = 3
    seg = Segment([spliceable] * Nexbody, 'N', 'C')
    Npairs_expected = Nexbody * Nexsite**2 * Npairs0
    assert seg.x2exit.shape == (Npairs_expected, 4, 4)
    assert seg.x2orgn.shape == (Npairs_expected, 4, 4)
    assert len(seg.entryresid) == Npairs_expected
    assert len(seg.exitresid) == Npairs_expected
    assert len(seg.bodyid) == Npairs_expected
    for i in range(Nexbody):
        assert i == seg.bodyid[0 + i * Npairs0 * Nexsite**2]
    assert np.all(seg.x2exit[..., 3, :3] == 0)
    assert np.all(seg.x2exit[..., 3, 3] == 1)
    for e2x, e2o, ir, jr in zip(seg.x2exit, seg.x2orgn,
                                seg.entryresid, seg.exitresid):
        assert np.allclose(stubs[ir - 1] @ e2o, np.eye(4), atol=1e-5)
        assert np.allclose(stubs[ir - 1] @ e2x, stubs[jr - 1], atol=1e-5)


@only_if_pyrosetta
def test_grow_cycle(c1pose):
    helix = Spliceable(c1pose, sites=[(1, 'N'), ('-4:', 'C')])
    segments = ([Segment([helix], exit='C'), ] +
                [Segment([helix], 'N', 'C')] * 3 +
                [Segment([helix], entry='N')])
    worms = grow(segments, Cyclic('C2', lever=20), thresh=20)
    assert 0.1411 < np.min(worms.scores) < 0.1412


@only_if_pyrosetta
def test_grow_cycle_thread_pool(c1pose):
    helix = Spliceable(c1pose, sites=[(1, 'N'), ('-4:', 'C')])
    segments = ([Segment([helix], exit='C'), ] +
                [Segment([helix], 'N', 'C')] * 3 +
                [Segment([helix], entry='N')])
    worms = grow(segments, Cyclic('C2', lever=20),
                 executor=ThreadPoolExecutor, max_workers=2)
    assert 0.1411 < np.min(worms.scores) < 0.1412
    assert np.sum(worms.scores < 0.1412) == 4


@pytest.mark.skipif('not HAVE_PYROSETTA_DISTRIBUTED')
def test_grow_cycle_process_pool(c1pose):
    helix = Spliceable(c1pose, sites=[(1, 'N'), ('-4:', 'C')])
    segments = ([Segment([helix], exit='C'), ] +
                [Segment([helix], 'N', 'C')] * 3 +
                [Segment([helix], entry='N')])
    worms = grow(segments, Cyclic('C2', lever=20),
                 executor=ProcessPoolExecutor, max_workers=2)
    assert 0.1411 < np.min(worms.scores) < 0.1412
    assert np.sum(worms.scores < 0.1412) == 4


@only_if_pyrosetta
def test_grow_errors(c1pose):
    nsplice = SpliceSite(sele=[1, 2, 3, 4, 5, 6], polarity='N')
    csplice = SpliceSite(sele=[13, ], polarity='C')
    spliceable1 = Spliceable(body=c1pose, sites=[nsplice, csplice])
    spliceable2 = Spliceable(body=c1pose, sites=[nsplice, csplice])
    spliceables = [spliceable1]
    segments = ([Segment(spliceables, exit='C'), ] +
                [Segment(spliceables, 'N', 'C'), ] * 3 +
                [Segment(spliceables, entry='N'), ])
    checkc3 = Cyclic('C2', from_seg=0, to_seg=-1)

    # make sure incorrect begin/end throws error
    with pytest.raises(ValueError):
        grow(segments[: 2], criteria=checkc3)
    with pytest.raises(ValueError):
        grow(segments[1:], criteria=checkc3)
    segments_polarity_mismatch = [
        Segment(spliceables, exit='C'),
        Segment(spliceables, entry='C'),
    ]
    with pytest.raises(ValueError):
        grow(segments_polarity_mismatch, criteria=checkc3)


@only_if_pyrosetta
def test_memsize(c1pose):
    helix = Spliceable(c1pose, sites=[((1, 2), 'N'), ('-2:', 'C')])
    segments = ([Segment([helix], exit='C'), ] +
                [Segment([helix], 'N', 'C')] * 3 +
                [Segment([helix], entry='N')])
    beg = 3
    for i in range(beg, 7):
        w1 = grow(segments, Cyclic('c2'), memsize=10**i, thresh=30)
        assert i == beg or len(w0.scores) == len(w1.scores)
        assert i == beg or np.allclose(w0.scores, w1.scores)
        w0 = w1


@only_if_pyrosetta
def test_pose_alignment_0(c1pose):
    helix = Spliceable(c1pose, sites=[(1, 'N'), ('-4:', 'C')])
    segments = ([Segment([helix], exit='C'), ] +
                [Segment([helix], 'N', 'C')] * 3 +
                [Segment([helix], entry='N')])
    w = grow(segments, Cyclic('c2'), thresh=1)
    assert len(w)
    print(w.indices)
    for i in range(4):
        assert tuple(w.indices[i]) in ((0, 2, 1, 2, 0), (2, 1, 2, 0, 0),
                                       (1, 2, 0, 2, 0), (2, 0, 2, 1, 0))
    pose = w.pose(0, align=1, end=1)
    assert util.no_overlapping_residues(pose)
    # vis.showme(pose)
    xyz0 = np.array([pose.residue(1).xyz(2)[i] for i in (0, 1, 2)] + [1])
    # resid 43 happens to be the symmetrically related one for this solution
    xyz1 = np.array([pose.residue(42).xyz(2)[i] for i in (0, 1, 2)] + [1])
    xyz1 = hrot([0, 0, 1], 180) @ xyz1
    assert np.sum((xyz1 - xyz0)**2) < 0.1


@only_if_pyrosetta
def test_last_body_same_as(c1pose):
    helix = Spliceable(c1pose, sites=[(1, 'N'), ('-4:', 'C')])
    segments = ([Segment([helix, helix], exit='C'), ] +
                [Segment([helix], 'N', 'C')] * 3 +
                [Segment([helix, helix], entry='N')])
    w = grow(segments, Cyclic('c2'), thresh=1)
    for i, s in zip(w.indices, w.scores):
        assert segments[0].bodyid[i[0]] == segments[-1].bodyid[i[-1]]
    assert len(w) == 8
    ref = [(1, 2, 0, 2, 0), (5, 2, 0, 2, 1), (2, 0, 2, 1, 0), (6, 0, 2, 1, 1),
           (0, 2, 1, 2, 0), (4, 2, 1, 2, 1), (2, 1, 2, 0, 0), (6, 1, 2, 0, 1)]
    for i in range(8):
        assert tuple(w.indices[i]) in ref


def test_reorder_spliced_as_N_to_C():
    Q = reorder_spliced_as_N_to_C

    with pytest.raises(ValueError): Q([[1], [1], [1]], 'NC')
    with pytest.raises(ValueError): Q([[1], [1], [1]], 'CN')
    with pytest.raises(ValueError): Q([[1, 1], [1], [1, 1]], 'CN')
    with pytest.raises(ValueError): Q([], 'CN')
    with pytest.raises(ValueError): Q([], '')
    with pytest.raises(ValueError): Q([[]], '')

    assert Q([[1]], '') == [[1]]
    assert Q([[1, 2]], '') == [[1], [2]]
    assert Q([[1], [2]], 'N') == [[1, 2]]
    assert Q([[1, 2], [3]], 'N') == [[1], [2, 3]]
    assert Q([[1, 2], [3, 4]], 'N') == [[1], [2, 3], [4]]
    assert Q([[1, 2, 3], [4, 5]], 'N') == [[1], [2], [3, 4], [5]]
    assert Q([[1], [2]], 'C') == [[2, 1]]
    assert Q([[1, 2], [3]], 'C') == [[1], [3, 2]]
    assert Q([[1, 2], [3, 4]], 'C') == [[1], [3, 2], [4]]
    assert Q([[1, 2, 3], [4, 5]], 'C') == [[1], [2], [4, 3], [5]]

    assert Q([[1], [2], [3]], 'NN') == [[1, 2, 3]]
    assert Q([[1], [2], [3, 4]], 'NN') == [[1, 2, 3], [4]]
    assert Q([[1], [2, 3], [4, 5]], 'NN') == [[1, 2], [3, 4], [5]]
    assert Q([[1, 2], [3, 4], [5, 6]], 'NN') == [[1], [2, 3], [4, 5], [6]]
    assert (Q([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'NN')
            == [[1], [2], [3, 4], [5], [6, 7], [8], [9]])
    assert (Q([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'CN')
            == [[1], [2], [4, 3], [5], [6, 7], [8], [9]])
    assert (Q([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'CC')
            == [[1], [2], [4, 3], [5], [7, 6], [8], [9]])
    assert (Q([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'NC')
            == [[1], [2], [3, 4], [5], [7, 6], [8], [9]])

    for n in range(10):
        x = [[i] for i in range(n + 1)]
        y = list(range(n + 1))
        assert Q(x, 'N' * n) == [y]
        assert Q(x, 'C' * n) == [y[::-1]]
        assert Q([[13, 14]] + x, 'N' + 'N' * n) == [[13], [14] + y]
        assert Q([[13, 14]] + x, 'C' + 'C' * n) == [[13], y[::-1] + [14]]
        assert (Q([[10, 11, 12]] + x + [[13, 14, 15]], 'N' + 'N' * n + 'N')
                == [[10], [11], [12] + y + [13], [14], [15]])
        assert (Q([[10, 11, 12]] + x + [[13, 14, 15]], 'C' + 'C' * n + 'C')
                == [[10], [11], [13] + y[::-1] + [12], [14], [15]])

    assert (Q([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]], 'NNN')
            == [[1], [2], [3, 4], [5], [6, 7], [8], [9, 0], [1], [2]])
    assert (Q([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]], 'CNN')
            == [[1], [2], [4, 3], [5], [6, 7], [8], [9, 0], [1], [2]])
    assert (Q([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]], 'NCN')
            == [[1], [2], [3, 4], [5], [7, 6], [8], [9, 0], [1], [2]])
    assert (Q([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]], 'NNC')
            == [[1], [2], [3, 4], [5], [6, 7], [8], [0, 9], [1], [2]])
    assert (Q([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]], 'NCC')
            == [[1], [2], [3, 4], [5], [7, 6], [8], [0, 9], [1], [2]])
    assert (Q([[1, 2, 3], [4, 5, 6], [11], [7, 8, 9], [0, 1, 2]], 'NCCC')
            == [[1], [2], [3, 4], [5], [7, 11, 6], [8], [0, 9], [1], [2]])
    assert (Q([[1, 2, 3], [4, 5, 6], [11], [12], [7, 8, 9], [0, 1, 2]], 'NCCCN')
            == [[1], [2], [3, 4], [5], [7, 12, 11, 6], [8], [9, 0], [1], [2]])
    assert (Q([[1, 2, 5, 5, 3], [4, 5, 6], [11], [12],
               [7, 8, 9], [0, 1, 2]], 'NCCCN')
            == [[1], [2], [5], [5], [3, 4], [5], [7, 12, 11, 6],
                [8], [9, 0], [1], [2]])


@only_if_pyrosetta
def test_make_pose_chains_dimer(c2pose):
    dimer = Spliceable(c2pose, sites=[('1,2:2', 'N'), ('2,3:3', 'N'),
                                      ('1,-4:-4', 'C'), ('2,-5:-5', 'C')])
    print(dimer)
    seq = dimer.body.sequence()[:12]

    dimerseg = Segment([dimer], 'N', '')
    enex, rest = dimerseg.make_pose_chains(0, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[1:], seq]
    assert [x[0].sequence() for x in rest] == []
    assert enex[-1][0] is dimer.chains[2]
    enex, rest = dimerseg.make_pose_chains(1, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[2:], seq]
    assert [x[0].sequence() for x in rest] == []
    assert enex[-1][0] is dimer.chains[1]

    dimerseg = Segment([dimer], 'C', '')
    enex, rest = dimerseg.make_pose_chains(0, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[:-3], seq]
    assert [x[0].sequence() for x in rest] == []
    assert enex[-1][0] is dimer.chains[2]
    enex, rest = dimerseg.make_pose_chains(1, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[:-4], seq]
    assert [x[0].sequence() for x in rest] == []
    assert enex[-1][0] is dimer.chains[1]

    dimerseg = Segment([dimer], '', 'N')
    enex, rest = dimerseg.make_pose_chains(0, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq, seq[1:]]
    assert [x[0].sequence() for x in rest] == []
    assert enex[0][0] is dimer.chains[2]
    enex, rest = dimerseg.make_pose_chains(1, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq, seq[2:]]
    assert [x[0].sequence() for x in rest] == []
    assert enex[0][0] is dimer.chains[1]

    dimerseg = Segment([dimer], 'N', 'N')
    enex, rest = dimerseg.make_pose_chains(0, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[1:], seq[2:]]
    assert [x[0].sequence() for x in rest] == []
    enex, rest = dimerseg.make_pose_chains(1, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[2:], seq[1:]]
    assert [x[0].sequence() for x in rest] == []
    with pytest.raises(IndexError):
        enex, rest = dimerseg.make_pose_chains(2, pad=(0, 1))

    dimerseg = Segment([dimer], 'N', 'C')
    enex, rest = dimerseg.make_pose_chains(0, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[1:-3]]
    assert [x[0].sequence() for x in rest] == [seq]
    assert rest[0][0] is dimer.chains[2]
    enex, rest = dimerseg.make_pose_chains(1, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[1:], seq[:-4]]
    assert [x[0].sequence() for x in rest] == []
    enex, rest = dimerseg.make_pose_chains(2, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[2:], seq[:-3]]
    assert [x[0].sequence() for x in rest] == []
    enex, rest = dimerseg.make_pose_chains(3, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[2:-4]]
    assert [x[0].sequence() for x in rest] == [seq]
    assert rest[0][0] is dimer.chains[1]
    with pytest.raises(IndexError):
        enex, rest = dimerseg.make_pose_chains(4, pad=(0, 1))


def residue_coords(p, ir, n=3):
    crd = (p.residue(ir).xyz(i) for i in range(1, n + 1))
    return np.stack([c.x, c.y, c.z, 1] for c in crd)


def residue_sym_err(p, ang, ir, jr, n=1, axis=[0, 0, 1], verbose=0):
    mxdist = 0
    for i in range(n):
        xyz0 = residue_coords(p, ir + i)
        xyz1 = residue_coords(p, jr + i)
        xyz3 = hrot(axis, ang) @ xyz1.T
        xyz4 = hrot(axis, -ang) @ xyz1.T
        if verbose:
            print(i, xyz0)
            print(i, xyz1)
            print(i, xyz3.T)
            print(i, xyz4.T)
            print()
        mxdist = max(mxdist, min(
            np.max(np.sum((xyz0 - xyz3.T)**2, axis=1)),
            np.max(np.sum((xyz0 - xyz4.T)**2, axis=1))))
    return np.sqrt(mxdist)


@only_if_pyrosetta
def test_multichain_match_reveres_pol(c1pose, c2pose):
    helix = Spliceable(
        c1pose, sites=[((1, 2, 3,), 'N'), ((9, 10, 11, 13), 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:1', 'N'), ('1,-1:', 'C'),
                                      ('2,:2', 'N'), ('2,-1:', 'C')])
    segments = [Segment([helix], exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([dimer], entry='N', exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([helix], entry='N'), ]
    wnc = grow(segments, Cyclic('C3', lever=20), thresh=1)
    assert len(wnc)
    assert wnc.scores[0] < 0.25

    segments = [Segment([helix], exit='N'),
                Segment([helix], entry='C', exit='N'),
                Segment([dimer], entry='C', exit='N'),
                Segment([helix], entry='C', exit='N'),
                Segment([helix], entry='C'), ]
    wcn = grow(segments, Cyclic('C3', lever=20), thresh=1)
    # assert residue_sym_err(wcn.pose(0), 120, 22, 35, 8) < 0.5
    # N-to-C and C-to-N construction should be same
    assert np.allclose(wnc.scores, wcn.scores, atol=1e-3)


@only_if_pyrosetta
def test_splicepoints(c1pose, c2pose, c3pose):
    helix = Spliceable(
        c1pose, sites=[((1, 2, 3,), 'N'), ((9, 10, 11, 13), 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:1', 'N'), ('1,-1:', 'C'),
                                      ('2,:2', 'N'), ('2,-1:', 'C')])
    segments = [Segment([helix], exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([dimer], entry='N', exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([helix], entry='N'), ]
    w = grow(segments, Cyclic('C3', lever=20), thresh=1)
    assert len(w) == 17
    assert w.scores[0] < 0.25
    assert w.splicepoints(0) == [11, 19, 27, 37]
    w.pose(0, cyclic_permute=0)
    assert w.splicepoints(0) == [10, 20, 42]

    helix = Spliceable(c1pose, [(':4', 'N'), ('-4:', 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:2', 'N'), ('1,-1:', 'C'),
                                      ('2,:2', 'N'), ('2,-1:', 'C')])
    trimer = Spliceable(c3pose, sites=[('1,:1', 'N'), ('1,-2:', 'C'),
                                       ('2,:2', 'N'), ('2,-2:', 'C'),
                                       ('3,:1', 'N'), ('3,-2:', 'C')])
    segments = [Segment([trimer], exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([dimer], entry='N')]
    w = grow(segments, D3(c2=-1, c3=0), thresh=1)
    assert len(w) == 90
    assert w.splicepoints(0) == [8, 16, 25, 34]

    actual_chains = list(w.pose(0, join=0).split_by_chain())
    for i, splice in enumerate(w.splices(0)):
        ib1, ic1, ir1, ib2, ic2, ir2, dr = splice
        pose1 = w.segments[i].spliceables[ib1].chains[ic1]
        pose2 = w.segments[i + 1].spliceables[ib2].chains[ic2]
        seq1 = str(util.subpose(pose1, 1, ir1 - 1).sequence())
        seq2 = str(util.subpose(pose2, ir2).sequence())
        # print(i, '1', seq1, str(actual_chains[i].sequence()))
        # print(i, '2', seq2, str(actual_chains[i + 1].sequence()))
        assert seq1.endswith(str(actual_chains[i].sequence()))
        assert seq2.startswith(str(actual_chains[i + 1].sequence()))


@only_if_pyrosetta
def test_cyclic_permute_beg_end(c1pose, c2pose):
    helix = Spliceable(
        c1pose, sites=[((1, 2, 3,), 'N'), ((9, 10, 11, 13), 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:1', 'N'), ('1,-1:', 'C'),
                                      ('2,:2', 'N'), ('2,-1:', 'C')])
    segments = [Segment([helix], exit='N'),
                Segment([helix], entry='C', exit='N'),
                Segment([dimer], entry='C', exit='N'),
                Segment([helix], entry='C', exit='N'),
                Segment([helix], entry='C'), ]
    w = grow(segments, Cyclic('C3', lever=50), thresh=1)
    # vis.showme(w.pose(0))
    p = w.pose(0, cyclic_permute=1)
    assert p.sequence() == 'YTAFLAAIPAINAAAAAAAGAAAAAGAAAAAAAGAAAAAFLAAIPAIN'
    assert p.chain(30) == 1
    assert util.no_overlapping_residues(p)

    segments = [Segment([helix], '_C'),
                Segment([helix], 'NC'),
                Segment([dimer], 'NC'),
                Segment([helix], 'NC'),
                Segment([helix], 'N_'), ]
    w = grow(segments, Cyclic('C3', lever=50), thresh=1)
    p = w.pose(0, cyclic_permute=1)
    assert p.sequence() == 'YTAFLAAIPAIAAAAAAAAAAAAAAGAAAAAAAGAAATAFLAAIPAIN'
    assert p.chain(len(p)) == 1
    assert util.no_overlapping_residues(p)
    # print(w.scores)
    # vis.showme(w.pose(0, cyclic_permute=0), name='reg')
    # print('------------------------')
    # vis.showme(w.pose(0, end=1, join=False), name='end')
    # print('------------------------')
    # vis.showme(w.pose(0, cyclic_permute=1), name='cp')
    # print('------------------------')
    # assert 0


@only_if_pyrosetta
def test_cyclic_permute_mid_end(c1pose, c2pose, c3hetpose):
    helix0 = Spliceable(c1pose, [([2], 'N'), ([11], "C")])
    helix = Spliceable(c1pose, [([1, 3, 4], 'N'), ([12, ], "C")])
    dimer = Spliceable(c2pose, sites=[('1,:1', 'N'), ('1,-1:', 'C'),
                                      ('2,:1', 'N'), ('2,-1:', 'C')])
    c3het = Spliceable(c3hetpose, sites=[
        ('1,2:2', 'N'), ('2,2:2', 'N'), ('3,2:2', 'N')])
    segments = [Segment([helix0], '_C'),
                Segment([helix0], 'NC'),
                Segment([helix0], 'NC'),
                Segment([c3het], 'NN'),
                Segment([helix], 'CN'),
                Segment([dimer], 'CC'),
                Segment([helix], 'NC'),
                Segment([helix], 'NC'),
                Segment([c3het], 'N_'), ]
    w = grow(segments, Cyclic(3, from_seg=3), thresh=1)
    assert len(w) > 0
    p, sc = w.sympose(0, score=True)
    assert sc < 4
    assert len(p) == 312
    assert p.chain(306) == 9
    assert util.no_overlapping_residues(p)
    # vis.showme(p)
    # p, pl = w.pose(0, cyclic_permute=0, end=0, cyclictrim=0,
    # make_chain_list=True)
    # vis.showme(p, name='full')
    # for i, p in enumerate(pl):
    # vis.showme(p, name='part%i' % i)

    # vis.showme(w.pose(0, cyclic_permute=0, end=0, cyclictrim=1), name='ct')
    # vis.showme(w.pose(0, cyclic_permute=0, end=0), name='cp0_end0')
    # vis.showme(w.pose(0, cyclic_permute=0, end=1), name='cp0_end1')
    # vis.showme(w.pose(0, cyclic_permute=1), name='cp1')
    # vis.showme(w.sympose(0, fullatom=True))

    # assert 0
    # for i in range(len(w)):
    # pose, score = w.sympose(i, score=1)
    # print(score)
    # assert len(w)
#
    # assert 0


@only_if_pyrosetta
def test_multichain_mixed_pol(c2pose, c3pose, c1pose):
    helix = Spliceable(c1pose, [(':4', 'N'), ((10, 12, 13), 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:2', 'N'), ('1,-1:', 'C'),
                                      ('2,:2', 'N'), ('2,-1:', 'C')])
    trimer = Spliceable(c3pose, sites=[('1,:1', 'N'), ('1,-2:', 'C'),
                                       ('2,:2', 'N'), ('2,-2:', 'C'),
                                       ('3,:1', 'N'), ('3,-2:', 'C')])
    segments = [Segment([helix], exit='C'),
                Segment([dimer], entry='N', exit='N'),
                Segment([helix], entry='C', exit='N'),
                Segment([trimer], entry='C', exit='C'),
                Segment([helix], entry='N')]
    w = grow(segments, Cyclic('C3'), thresh=1)
    assert len(w) == 24
    p = w.pose(0, end=True, cyclic_permute=0)
    assert util.no_overlapping_residues(p)
    # vis.show_with_axis(w, 0)
    # vis.showme(p)

    assert 1 > residue_sym_err(p, 120, 2, 62, 7)


@only_if_pyrosetta
def test_multichain_db(c2pose, c1pose):
    helix = Spliceable(c1pose, [(':4', 'N'), ('-4:', "C")])
    dimer = Spliceable(c2pose, sites=[('1,-1:', 'C'), ('2,-1:', 'C')])
    segments = [Segment([helix], exit='N'),
                Segment([dimer], entry='C', exit='C'),
                Segment([helix], entry='N')]
    with pytest.raises(ValueError):
        w = grow(segments, Cyclic('C4'), thresh=20)


@only_if_pyrosetta
def test_D3(c2pose, c3pose, c1pose):
    helix = Spliceable(c1pose, [(':4', 'N'), ('-4:', 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:2', 'N'), ('1,-1:', 'C'),
                                      ('2,:2', 'N'), ('2,-1:', 'C')])
    trimer = Spliceable(c3pose, sites=[('1,:1', 'N'), ('1,-2:', 'C'),
                                       ('2,:2', 'N'), ('2,-2:', 'C'),
                                       ('3,:1', 'N'), ('3,-2:', 'C')])
    segments = [Segment([trimer], exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([dimer], entry='N')]
    w = grow(segments, D3(c2=-1, c3=0), thresh=1)
    # print(w.scores)
    # show_with_z_axes(w, 0)
    p = w.pose(0, only_connected=0)
    assert util.no_overlapping_residues(p)
    # print(len(p))

    assert 1 > residue_sym_err(p, 180, 53, 65, 6, axis=[1, 0, 0])
    assert 1 > residue_sym_err(p, 120, 1, 10, 6, axis=[0, 0, 1])
    # assert 0
    segments = [Segment([dimer], exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([trimer], entry='N')]
    w = grow(segments, D3(c2=0, c3=-1), thresh=1)
    # print(w.scores)
    # show_with_z_axes(w, 0)
    p = w.pose(4, only_connected=0)
    assert util.no_overlapping_residues(p)
    # vis.showme(p)
    assert 1 > residue_sym_err(p, 180, 1, 13, 6, axis=[1, 0, 0])
    assert 1 > residue_sym_err(p, 120, 56, 65, 6, axis=[0, 0, 1])


@only_if_pyrosetta
def test_tet(c2pose, c3pose, c1pose):
    helix = Spliceable(c1pose, [(':1', 'N'), ('-4:', 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:2', 'N'), ('1,-1:', 'C'), ])
    trimer = Spliceable(c3pose, sites=[('1,:1', 'N'), ('1,-2:', 'C'), ])
    segments = ([Segment([dimer], exit='C')] +
                [Segment([helix], entry='N', exit='C')] * 5 +
                [Segment([trimer], entry='N')])
    w = grow(segments, Tetrahedral(c3=-1, c2=0), thresh=2)
    assert len(w)
    p = w.pose(3, only_connected=0)
    assert util.no_overlapping_residues(p)
    assert 2.5 > residue_sym_err(p, 120, 86, 95, 6, axis=[1, 1, 1])
    assert 2.5 > residue_sym_err(p, 180, 2, 14, 6, axis=[1, 0, 0])


@only_if_pyrosetta
def test_tet33(c2pose, c3pose, c1pose):
    helix = Spliceable(c1pose, [(':1', 'N'), ('-4:', 'C')])
    trimer = Spliceable(c3pose, sites=[('1,:1', 'N'), ('1,-2:', 'C'), ])
    segments = ([Segment([trimer], exit='C')] +
                [Segment([helix], entry='N', exit='C')] * 5 +
                [Segment([trimer], entry='N')])
    w = grow(segments, Tetrahedral(c3=-1, c3b=0), thresh=2)
    assert len(w) == 3
    p = w.pose(0, only_connected=0)
    assert util.no_overlapping_residues(p)
    assert 2.5 > residue_sym_err(p, 120, 2, 20, 6, axis=[1, 1, -1])
    assert 2.5 > residue_sym_err(p, 120, 87, 96, 6, axis=[1, 1, 1])


@only_if_pyrosetta
def test_oct(c2pose, c3pose, c4pose, c1pose):
    helix = Spliceable(c1pose, [(':1', 'N'), ('-4:', 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:2', 'N'), ('1,-1:', 'C'), ])
    trimer = Spliceable(c3pose, sites=[('1,:1', 'N'), ('1,-2:', 'C'), ])
    tetramer = Spliceable(c4pose, sites=[('1,:1', 'N'), ('1,-2:', 'C'), ])
    segments = ([Segment([dimer], exit='C')] +
                [Segment([helix], entry='N', exit='C')] * 5 +
                [Segment([trimer], entry='N')])
    w = grow(segments, Octahedral(c3=-1, c2=0), thresh=1)
    assert len(w) == 1
    p = w.pose(0, only_connected=0)
    assert util.no_overlapping_residues(p)
    assert 1 > residue_sym_err(p, 120, 85, 94, 6, axis=[1, 1, 1])
    assert 1 > residue_sym_err(p, 180, 1, 13, 6, axis=[1, 1, 0])

    segments = ([Segment([tetramer], exit='C')] +
                [Segment([helix], entry='N', exit='C')] * 5 +
                [Segment([dimer], entry='N')])
    w = grow(segments, Octahedral(c2=-1, c4=0), thresh=1)
    assert len(w) == 5
    assert np.allclose(w.indices, np.array([[0, 1, 1, 2, 0, 2, 1],
                                            [1, 0, 2, 3, 1, 0, 1],
                                            [1, 0, 0, 0, 3, 2, 0],
                                            [0, 2, 0, 0, 1, 2, 0],
                                            [1, 1, 2, 1, 1, 2, 0]]))
    p = w.pose(0, only_connected=0)
    assert p.sequence() == ('AIAAALAAIAAIAAALAAIAAIAAALAAIAAIAAALAAAAAAAAAAGA'
                            + 'AAAAAAAAGAAAAAAAAAGAAAAAAAAAAGAAAAAAAAGAATAFLA'
                            + 'AIPAINYTAFLAAIPAIN')
    assert util.no_overlapping_residues(p)
    # from socket import gethostname
    # p.dump_pdb(gethostname() + '.pdb')
    # assert np.allclose(p.residue(1).xyz('CA')[0], 33.0786722948)
    assert 1 > residue_sym_err(p, 90, 1, 31, 6, axis=[1, 0, 0], verbose=0)
    assert 1 > residue_sym_err(p, 180, 92, 104, 6, axis=[1, 1, 0], verbose=0)
    # assert 0


@only_if_pyrosetta
def test_icos(c2pose, c3pose, c1pose):
    helix = Spliceable(c1pose, [(':1', 'N'), ('-4:', 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:2', 'N'), ('1,-1:', 'C'), ])
    trimer = Spliceable(c3pose, sites=[('1,:1', 'N'), ('1,-2:', 'C'), ])
    segments = ([Segment([dimer], exit='C')] +
                [Segment([helix], entry='N', exit='C')] * 5 +
                [Segment([trimer], entry='N')])
    w = grow(segments, Icosahedral(c3=-1, c2=0), thresh=2)
    assert len(w) == 3
    p = w.pose(2, only_connected=0)
    assert util.no_overlapping_residues(p)
    # vis.showme(p)
    assert 2 > residue_sym_err(p, 120, 90, 99, 6, axis=IA[3])
    assert 2 > residue_sym_err(p, 180, 2, 14, 6, axis=IA[2])


@only_if_pyrosetta
def test_score0_sym(c2pose, c3pose, c1pose):
    helix = Spliceable(c1pose, [(':1', 'N'), ('-4:', 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:2', 'N'), ('1,-1:', 'C'), ])
    trimer = Spliceable(c3pose, sites=[('1,:1', 'N'), ('1,-2:', 'C'), ])
    segments = ([Segment([dimer], exit='C')] +
                [Segment([helix], entry='N', exit='C')] * 4 +
                [Segment([trimer], entry='N')])
    w = grow(segments, D3(c3=-1, c2=0), thresh=2)
    assert len(w) == 3
    i, err, pose, score0 = w[1]
    # vis.showme(w.pose(1, fullatom=True))
    # show_with_z_axes(w, 1)
    assert 22.488 < score0 < 22.4881
    assert util.no_overlapping_residues(pose)

    t = time.time()
    ps1 = w.sympose(range(3), score=1)
    t = time.time() - t
    print(t)

    if hasattr(pose, '__getstate__'):
        t = time.time()
        ps2 = w.sympose(range(3), score=1, parallel=1)
        t = time.time() - t
        print(t)
        assert np.allclose([x[1] for x in ps1], [x[1] for x in ps2])


@only_if_pyrosetta
def test_chunk_speed(c2pose, c3pose, c1pose):
    helix = Spliceable(c1pose, [(':1', 'N'), ('-2:', 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:2', 'N'), ('1,-1:', 'C'), ])
    trimer = Spliceable(c3pose, sites=[('1,:1', 'N'), ('1,-2:', 'C'), ])
    nseg = 11
    segments = ([Segment([dimer], exit='C')] +
                [Segment([helix], entry='N', exit='C')] * (nseg - 2) +
                [Segment([trimer], entry='N')])
    # w = grow(segments, Tetrahedral(c3=-1, c2=0), thresh=5)
    t1 = time.time()
    w1 = grow(segments, Octahedral(c3=-1, c2=0), thresh=1, memsize=0)
    t1 = time.time() - t1
    t2 = time.time()
    w2 = grow(segments, Octahedral(c3=-1, c2=0), thresh=1, memsize=1e7)
    t2 = time.time() - t2

    print('chunksize', w1.detail['chunksize'], 'time', t1)
    print('chunksize', w2.detail['chunksize'], 'time', t2)
    print('speedup:', t1 / t2)

    assert t1 / t2 > 10.0  # conservative, but still sketchy...


@only_if_pyrosetta
def test_splice_compatibility_check(c1pose, c2pose):
    helix = Spliceable(c1pose, [(':1', 'N'), ('-2:', 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:2', 'N'), ('2,:2', 'N'), ])
    segments = [Segment([helix], '_C'),
                Segment([dimer], 'NN'),
                Segment([helix], 'C_'), ]
    with pytest.raises(ValueError):
        w = grow(segments, Cyclic(), thresh=1)


@only_if_pyrosetta
def test_invalid_splices_seg_too_small(c1pose):
    helix = Spliceable(c1pose, [('8:8', 'N'), ('7:7', 'C')])
    with pytest.raises(ValueError):
        segments = [Segment([helix], '_C'),
                    Segment([helix], 'NC'),
                    Segment([helix], 'N_')]

    helix = Spliceable(c1pose, [('7:8', 'N'), ('7:8', 'C')])
    segments = [Segment([helix], '_C'),
                Segment([helix], 'NC'),
                Segment([helix], 'N_')]
    w = grow(segments, Cyclic('C3'), thresh=9e9)
    assert len(w) == 12

    helix = Spliceable(c1pose, [('7:8', 'N'), ('7:8', 'C')], min_seg_len=2)
    segments = [Segment([helix], '_C'),
                Segment([helix], 'NC'),
                Segment([helix], 'N_')]
    w = grow(segments, Cyclic('C3'), thresh=9e9)
    assert len(w) == 4


@only_if_pyrosetta
def test_invalid_splices_site_overlap_2(c1pose, c2pose):
    helix = Spliceable(c1pose, [(':1', 'N'), ('-1:', 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:1', 'N'), ('2,:1', 'N'),
                                      ('1,-1:', 'C'), ('2,-1:', 'C'), ])
    segments = [Segment([helix], '_C'),
                Segment([dimer], 'NN'),
                Segment([helix], 'CN'),
                Segment([dimer], 'CC'),
                Segment([helix], 'N_'), ]
    w = grow(segments, Cyclic(3), thresh=9e9)
    assert len(w) == 4
    for i in range(len(w)):
        assert (w.segments[1].entrysiteid[w.indices[i, 1]] !=
                w.segments[1].exitsiteid[w.indices[i, 1]])
        assert (w.segments[3].entrysiteid[w.indices[i, 3]] !=
                w.segments[3].exitsiteid[w.indices[i, 3]])


@only_if_pyrosetta
def test_invalid_splices_site_overlap_3(c1pose, c3pose):
    helix = Spliceable(c1pose, [(':1', 'N'), ('-1:', 'C')])
    trimer = Spliceable(c3pose, sites=[('1,:1', 'N'), ('1,-1:', 'C'),
                                       ('2,:1', 'N'), ('2,-1:', 'C'),
                                       ('3,:1', 'N'), ('3,-1:', 'C'), ])
    segments = [Segment([helix], '_C'),
                Segment([trimer], 'NN'),
                Segment([helix], 'CN'),
                Segment([trimer], 'CC'),
                Segment([helix], 'NC'),
                Segment([trimer], 'N_'), ]
    w = grow(segments, Cyclic(3, from_seg=1), thresh=9e9)
    assert len(w)
    for i in range(len(w)):
        assert (w.segments[1].entrysiteid[w.indices[i, 1]] !=
                w.segments[1].exitsiteid[w.indices[i, 1]])
        assert (w.segments[1].entrysiteid[w.indices[i, 1]] !=
                w.segments[5].entrysiteid[w.indices[i, 5]])
        assert (w.segments[1].exitsiteid[w.indices[i, 1]] !=
                w.segments[5].entrysiteid[w.indices[i, 5]])


@pytest.mark.skip  # if('not HAVE_PYROSETTA')
def test_origin_seg(c1pose, c2pose, c3pose):
    helix = Spliceable(c1pose, [(':1', 'N'), ('-8:', 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:3', 'N'), ('1,-3:', 'C'),
                                      ('2,:3', 'N'), ('2,-3:', 'C'), ])
    trimer = Spliceable(c3pose, sites=[('1,:3', 'N'), ('1,-3:', 'C'),
                                       ('2,:3', 'N'), ('2,-3:', 'C'),
                                       ('3,:3', 'N'), ('3,-3:', 'C'), ])
    segments = [Segment([trimer], '_C'),  # origin_seg
                Segment([helix], 'NC'),
                Segment([trimer], 'NN'),  # from_seg
                Segment([helix], 'CN'),
                Segment([dimer], 'CC'),
                Segment([helix], 'NC'),
                Segment([trimer], 'N_'), ]  # to_seg
    w = grow(segments, Cyclic(3, from_seg=2, origin_seg=0), thresh=10)
    # executor=ProcessPoolExecutor, max_workers=8)
    assert len(w) > 0
    print(w.scores[:10])
    vis.showme(w.pose(0, join=False))
    # assert 0


@only_if_pyrosetta
def test_provenance(c1pose):
    sites = [(':1', 'N'), ('-4:', 'C')]
    segments = [Segment([Spliceable(c1pose.clone(), sites)], '_C'),
                Segment([Spliceable(c1pose.clone(), sites)], 'NC'),
                Segment([Spliceable(c1pose.clone(), sites)], 'NC'),
                Segment([Spliceable(c1pose.clone(), sites)], 'NC'),
                Segment([Spliceable(c1pose.clone(), sites)], 'NC'),
                Segment([Spliceable(c1pose.clone(), sites)], 'NC'),
                Segment([Spliceable(c1pose.clone(), sites)], 'NC'),
                Segment([Spliceable(c1pose.clone(), sites)], 'N_')]
    w = grow(segments, Cyclic(6), thresh=2, expert=True)
    assert len(w)
    for i in range(len(w)):
        # pose, score, srcpose, srcres = w.sympose(
            # i, score=True, provenance=True)
        pose, prov = w.pose(i, provenance=True)

        for i, prv in enumerate(prov):
            lb, ub, src_pose, src_lb, src_ub = prv
            assert src_pose is segments[i].spliceables[0].body
            assert src_pose is not c1pose
            srcseq = src_pose.sequence()[src_lb - 1:src_ub]
            seq = pose.sequence()[lb - 1:ub]
            assert srcseq == seq
        assert len(prov) == len(segments) - 1


@only_if_pyrosetta
def test_extra_chain_handling_cyclic(c1pose, c2pose, c3hetpose):
    helix = Spliceable(c1pose, [(':1', 'N'), ('-4:', 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:3', 'N'), ('1,-3:', 'C')])
    trimer = Spliceable(c3hetpose, sites=[('1,:3', 'N'), ('2,-3:', 'C')])

    segments = [Segment([helix], '_C'),
                Segment([dimer], 'NC'),
                Segment([helix], 'N_'), ]
    w = grow(segments, Cyclic(9), thresh=3)
    assert len(w) == 1
    assert tuple(w.indices[0]) == (2, 7, 0)
    p, prov = w.pose(0, provenance=1, only_connected=0)
    assert len(prov) == 3
    assert prov[0] == (1, 11, c1pose, 1, 11)
    assert prov[1] == (12, 19, c2pose, 3, 10)
    assert prov[2] == (21, 32, c2pose, 13, 24)
    p, prov = w.pose(0, provenance=1, only_connected=1)
    assert len(prov) == 2
    assert prov[0] == (1, 11, c1pose, 1, 11)
    assert prov[1] == (12, 19, c2pose, 3, 10)
    p, prov = w.pose(0, provenance=1, only_connected='auto')
    assert len(prov) == 3
    assert prov[0] == (1, 11, c1pose, 1, 11)
    assert prov[1] == (12, 19, c2pose, 3, 10)
    assert prov[2] == (21, 32, c2pose, 13, 24)

    segments = [Segment([helix], '_C'),
                Segment([trimer], 'NC'),
                Segment([helix], 'N_'), ]
    w = grow(segments, Cyclic(6), thresh=3)
    # vis.showme(w.pose(0))
    assert len(w) == 1
    assert tuple(w.indices[0]) == (3, 7, 0)
    p, prov = w.pose(0, provenance=1, only_connected=0)
    assert len(prov) == 4
    assert prov[0] == (1, 7, c3hetpose, 10, 16)
    assert prov[1] == (8, 19, c1pose, 1, 12)
    assert prov[2] == (20, 26, c3hetpose, 3, 9)
    assert prov[3] == (27, 35, c3hetpose, 19, 27)
    p, prov = w.pose(0, provenance=1, only_connected=1)
    assert len(prov) == 3
    assert prov[0] == (1, 7, c3hetpose, 10, 16)
    assert prov[1] == (8, 19, c1pose, 1, 12)
    assert prov[2] == (20, 26, c3hetpose, 3, 9)
    # assert prov[3] == (27, 35, c3hetpose, 19, 27)
    p, prov = w.pose(0, provenance=1, only_connected='auto')
    assert len(prov) == 4
    assert prov[0] == (1, 7, c3hetpose, 10, 16)
    assert prov[1] == (8, 19, c1pose, 1, 12)
    assert prov[2] == (20, 26, c3hetpose, 3, 9)
    assert prov[3] == (27, 35, c3hetpose, 19, 27)


@only_if_pyrosetta
def test_extra_chain_handling_noncyclic(c1pose, c2pose, c3pose, c3hetpose):
    helix = Spliceable(c1pose, [(':4', 'N'), ('-4:', 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:1', 'N'), ('1,-1:', 'C')])
    trimer = Spliceable(c3pose, sites=[('1,:1', 'N'), ('1,-2:', 'C')])
    hettri = Spliceable(c3hetpose, sites=[('1,:1', 'N'), ('1,-1:', 'C')])
    segments = [Segment([trimer], exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([hettri], entry='N', exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([dimer], entry='N')]
    w = grow(segments, D3(c2=-1, c3=0), thresh=1)
    # vis.showme(w.sympose(0, fullatom=1))
    assert len(w) == 4
    assert w.pose(0, only_connected='auto').num_chains() == 3
    assert w.pose(0, only_connected=0).num_chains() == 6
    assert w.pose(0, only_connected=1).num_chains() == 1

    hettri = Spliceable(c3hetpose, sites=[('1,:1', 'N'), ('2,-1:', 'C')])
    segments = [Segment([trimer], exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([hettri], entry='N', exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([dimer], entry='N')]
    w = grow(segments, D3(c2=-1, c3=0), thresh=1)
    assert len(w) == 1
    assert w.pose(0, only_connected='auto').num_chains() == 3
    assert w.pose(0, only_connected=0).num_chains() == 6
    assert w.pose(0, only_connected=1).num_chains() == 2


@only_if_pyrosetta
def test_max_results(c1pose, c2pose, c3pose):
    helix = Spliceable(c1pose, [(':4', 'N'), ('-4:', 'C')])
    dimer = Spliceable(c2pose, sites=[('1,:2', 'N'), ('1,-1:', 'C'),
                                      ('2,:2', 'N'), ('2,-1:', 'C')])
    trimer = Spliceable(c3pose, sites=[('1,:1', 'N'), ('1,-2:', 'C'),
                                       ('2,:2', 'N'), ('2,-2:', 'C'),
                                       ('3,:1', 'N'), ('3,-2:', 'C')])
    segments = [Segment([trimer], exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([helix], entry='N', exit='C'),
                Segment([dimer], entry='N')]
    wref = grow(segments, D3(c2=-1, c3=0), thresh=1)
    assert len(wref) == 90

    s = wref.scores[:]
    s.sort()
    i = np.argmin(s[1:] - s[:-1])

    wtst = grow(segments, D3(c2=-1, c3=0), thresh=1, max_results=90)
    assert len(wtst) == 90

    assert np.all(wref.indices == wtst.indices)


@only_if_pyrosetta
def test_chunk_speed(c2pose, c3pose, c1pose):
    helix = Spliceable(c1pose, [(':1', 'N'), ('-4:', 'C')])
    nseg = 39
    segments = ([Segment([helix], exit='C')] +
                [Segment([helix], entry='N', exit='C')] * (nseg - 2) +
                [Segment([helix], entry='N')])
    with pytest.raises(ValueError):
        grow(segments, Octahedral(c3=-1, c2=0), thresh=1, max_samples=1000000)


@only_if_pyrosetta
def test_NullCriteria(c1pose):
    helix = Spliceable(c1pose, [(':4', 'N'), ('-4:', 'C')])
    segments = [Segment([helix], '_C'),
                Segment([helix], 'N_')]
    results = grow(segments, NullCriteria())
    assert len(results) == 16
    # vis.showme(results.pose(0))
