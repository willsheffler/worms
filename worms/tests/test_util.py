from worms import util
import json
import itertools as it
import pytest
try:
    import pyrosetta
    HAVE_PYROSETTA = True
except ImportError:
    HAVE_PYROSETTA = False


@pytest.mark.skip()
def test_infer_symmetry(c1pose, c2pose, c3pose, c3hetpose, c6pose):
    print(c3pose)
    assert 0


def test_MultiRange():
    mr = util.MultiRange([2, 3, 4, 2, 3])
    prod = it.product(*[range(n) for n in mr.nside])
    for i, tup in enumerate(prod):
        assert tup == mr[i]
    assert i + 1 == len(mr)


@pytest.mark.skip()
def test_remove_dicts():
    jd = json.load(open(test_db_files[0]))[0]
    ji = dicts_to_items(jd)
    assert jd == items_to_dicts(ji)
    assert ji == dicts_to_items(jd)
    assert isinstance(jd, dict)
    assert isinstance(ji, list)
    assert isinstance(ji[0], tuple)
    assert len(ji[0]) is 2
