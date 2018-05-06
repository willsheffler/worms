from functools import partial
import pytest
from worms.pose_contortions import make_pose_chains_from_seg
from worms.segments import Spliceable, Segment
from worms.tests import only_if_pyrosetta


@only_if_pyrosetta
def test_make_pose_chains_dimer(c2pose):

    make_chains0 = make_pose_chains_from_seg

    dimer = Spliceable(
        c2pose,
        sites=[
            ('1,2:2', 'N'),
            ('2,3:3', 'N'),
            ('1,-4:-4', 'C'),
            ('2,-5:-5', 'C'),
        ])

    seq = dimer.body.sequence()[:12]

    make_chains = partial(make_chains0, Segment([dimer], 'N', None))
    enex, rest = make_chains(0, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[1:], seq]
    assert [x[0].sequence() for x in rest] == []
    assert enex[-1][0] is dimer.chains[2]
    enex, rest = make_chains(1, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[2:], seq]
    assert [x[0].sequence() for x in rest] == []
    assert enex[-1][0] is dimer.chains[1]

    make_chains = partial(make_chains0, Segment([dimer], 'C', None))
    enex, rest = make_chains(0, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[:-3], seq]
    assert [x[0].sequence() for x in rest] == []
    assert enex[-1][0] is dimer.chains[2]
    enex, rest = make_chains(1, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[:-4], seq]
    assert [x[0].sequence() for x in rest] == []
    assert enex[-1][0] is dimer.chains[1]

    make_chains = partial(make_chains0, Segment([dimer], None, 'N'))
    enex, rest = make_chains(0, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq, seq[1:]]
    assert [x[0].sequence() for x in rest] == []
    assert enex[0][0] is dimer.chains[2]
    enex, rest = make_chains(1, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq, seq[2:]]
    assert [x[0].sequence() for x in rest] == []
    assert enex[0][0] is dimer.chains[1]

    make_chains = partial(make_chains0, Segment([dimer], 'N', 'N'))
    enex, rest = make_chains(0, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[1:], seq[2:]]
    assert [x[0].sequence() for x in rest] == []
    enex, rest = make_chains(1, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[2:], seq[1:]]
    assert [x[0].sequence() for x in rest] == []
    with pytest.raises(IndexError):
        enex, rest = make_chains(2, pad=(0, 1))

    make_chains = partial(make_chains0, Segment([dimer], 'N', 'C'))
    enex, rest = make_chains(0, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[1:-3]]
    assert [x[0].sequence() for x in rest] == [seq]
    assert rest[0][0] is dimer.chains[2]
    enex, rest = make_chains(1, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[1:], seq[:-4]]
    assert [x[0].sequence() for x in rest] == []
    enex, rest = make_chains(2, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[2:], seq[:-3]]
    assert [x[0].sequence() for x in rest] == []
    enex, rest = make_chains(3, pad=(0, 1))
    assert [x[0].sequence() for x in enex] == [seq[2:-4]]
    assert [x[0].sequence() for x in rest] == [seq]
    assert rest[0][0] is dimer.chains[1]
    with pytest.raises(IndexError):
        enex, rest = make_chains(4, pad=(0, 1))
