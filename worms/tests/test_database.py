import pytest
from worms.database import *
import logging
import json
from worms.util import InProcessExecutor
from pprint import pprint
from os.path import dirname
import numba as nb

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

test_db_files = [
    'c6_database.json', 'HBRP_Cx_database.json',
    'HFuse_Cx_database.20180219.json',
    'HFuse_het_2chain_2arm_database.ZCON-103_2.20180406.json',
    'HFuse_het_2chain_2arm_database.ZCON-112_2.20180406.json',
    'HFuse_het_2chain_2arm_database.ZCON-127_2.20180406.json',
    'HFuse_het_2chain_2arm_database.ZCON-13_2.20180406.json',
    'HFuse_het_2chain_2arm_database.ZCON-15_2.20180406.json',
    'HFuse_het_2chain_2arm_database.ZCON-34_2.20180406.json',
    'HFuse_het_2chain_2arm_database.ZCON-37_2.20180406.json',
    'HFuse_het_2chain_2arm_database.ZCON-39_2.20180406.json',
    'HFuse_het_2chain_2arm_database.ZCON-9_2.20180406.json',
    'HFuse_het_3chain_2arm_database.Sh13_3.20180406.json',
    'HFuse_het_3chain_2arm_database.Sh13_3.20180416.json',
    'HFuse_het_3chain_2arm_database.Sh29_3.20180406.json',
    'HFuse_het_3chain_2arm_database.Sh29_3.20180416.json',
    'HFuse_het_3chain_2arm_database.Sh34_3.20180416.json',
    'HFuse_het_3chain_2arm_database.Sh3e_3.20180406.json',
    'HFuse_het_3chain_3arm_database.Sh13_3.20180406.json',
    'HFuse_het_3chain_3arm_database.Sh13_3.20180416.json',
    'HFuse_het_3chain_3arm_database.Sh29_3.20180406.json',
    'HFuse_het_3chain_3arm_database.Sh29_3.20180416.json',
    'HFuse_het_3chain_3arm_database.Sh34_3.20180416.json',
    'HFuse_het_3chain_3arm_database.Sh3e_3.20180406.json',
    'master_database_generation2.json'
]
test_db_files = [dirname(__file__) + '/../data/' + f for f in test_db_files]


@only_if_pyrosetta
def test_database_simple(tmpdir, caplog):
    pp = PDBPile(bakerdb_files=test_db_files)
    assert len(pp.query('C3_N')) == 213
    assert len(pp.query('Het_C3_C')) == 30
    assert len(pp.query('C2_N')) == 11
    assert len(pp.query('Het:NN')) == 9805
    assert len(pp.query('Het:NNX')) == 7501
    assert len(pp.query('Het:NNY')) == 2304
    assert 7501 + 2304 == 9805
    # pp.load_cached_coord_into_memory(pp.query('C3_N'))
    # assert len(pp.cache) == 213


@only_if_pyrosetta
def test_make_pdbdat(datadir, tmpdir):
    pp = PDBPile(
        cachedir=str(tmpdir),
        bakerdb_files=[os.path.join(datadir, 'test_db_file.json')],
        metaonly=False,
        read_new_pdbs=True)

    assert [len(pp.pose(k)) for k in pp.query('all')] == [
        8, 27, 36, 13, 9, 7, 13, 27, 40, 27, 24, 35
    ]

    keys = sorted(pp.cache.keys())

    assert np.allclose(pp.cache[keys[0]].connect_resids(0), np.array([0]))
    assert np.allclose(pp.cache[keys[0]].connect_resids(1), np.array([12]))
    assert np.allclose(pp.cache[keys[1]].connect_resids(0), np.array([0]))
    assert np.allclose(pp.cache[keys[1]].connect_resids(1), np.array([11]))
    assert np.allclose(pp.cache[keys[2]].connect_resids(0), np.array([0]))
    assert np.allclose(pp.cache[keys[2]].connect_resids(1), np.array([8]))
    assert np.allclose(pp.cache[keys[3]].connect_resids(0), np.array([0]))
    assert np.allclose(pp.cache[keys[3]].connect_resids(1), np.array([8]))
    assert np.allclose(pp.cache[keys[3]].connect_resids(2), np.array([9]))
    assert np.allclose(pp.cache[keys[3]].connect_resids(3), np.array([17]))
    assert np.allclose(pp.cache[keys[3]].connect_resids(4), np.array([18]))
    assert np.allclose(pp.cache[keys[3]].connect_resids(5), np.array([26]))
    assert np.allclose(pp.cache[keys[4]].connect_resids(0), np.array([0]))
    assert np.allclose(pp.cache[keys[4]].connect_resids(1), np.array([8]))
    assert np.allclose(pp.cache[keys[4]].connect_resids(2), np.array([9]))
    assert np.allclose(pp.cache[keys[4]].connect_resids(3), np.array([17]))
    assert np.allclose(pp.cache[keys[4]].connect_resids(4), np.array([18]))
    assert np.allclose(pp.cache[keys[4]].connect_resids(5), np.array([26]))
    assert np.allclose(pp.cache[keys[5]].connect_resids(0), np.array([0]))
    assert np.allclose(pp.cache[keys[5]].connect_resids(1), np.array([9]))
    assert np.allclose(pp.cache[keys[6]].connect_resids(0), np.array([0]))
    assert np.allclose(pp.cache[keys[6]].connect_resids(1), np.array([6]))
    assert np.allclose(pp.cache[keys[7]].connect_resids(0), np.array([0]))
    assert np.allclose(pp.cache[keys[7]].connect_resids(1), np.array([5]))
    assert np.allclose(pp.cache[keys[8]].connect_resids(0), np.array([0]))
    assert np.allclose(pp.cache[keys[8]].connect_resids(1),
                       np.array([6, 7, 8, 9, 10, 11, 12]))
    assert np.allclose(pp.cache[keys[9]].connect_resids(0), np.array([0]))
    assert np.allclose(pp.cache[keys[9]].connect_resids(1), np.array([7]))
    assert np.allclose(pp.cache[keys[10]].connect_resids(0), np.array([0]))
    assert np.allclose(pp.cache[keys[10]].connect_resids(1), np.array([6]))
    assert np.allclose(pp.cache[keys[11]].connect_resids(0), np.array([0]))
    assert np.allclose(pp.cache[keys[11]].connect_resids(1), np.array([8]))


def test_make_connections_array1():
    entries = [
        {"direction": "N", "chain": 1, "residues": "1,:1"},
        {"direction": "C", "chain": 1, "residues": "1,-7:"},
        {"direction": "N", "chain": 2, "residues": "2,:2"},
        {"direction": "C", "chain": 2, "residues": "2,-2:"},
        {"direction": "N", "chain": 3, "residues": "3,:3"},
        {"direction": "C", "chain": 3, "residues": "3,-3:"},
        ] # yapf: disable
    chain_bounds = [(0, 9), (9, 18), (18, 27)]
    a = make_connections_array(entries, chain_bounds)
    b = np.array([
        [0, 3, 0, -1, -1, -1, -1, -1, -1],
        [1, 9, 2, 3, 4, 5, 6, 7, 8],
        [0, 4, 9, 10, -1, -1, -1, -1, -1],
        [1, 4, 16, 17, -1, -1, -1, -1, -1],
        [0, 5, 18, 19, 20, -1, -1, -1, -1],
        [1, 5, 24, 25, 26, -1, -1, -1, -1],
    ])
    assert np.all(a == b)
