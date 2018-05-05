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
def test_make_pdbdat(tmpdir, datadir):
    pp = PDBPile(
        cachedir=str(tmpdir),
        bakerdb_files=[os.path.join(datadir, 'test_db_file.json')],
        metaonly=False,
        read_new_pdbs=True)

    # print([len(pp.pose(k)) for k in pp.query('all')])
    assert [len(pp.pose(k)) for k in pp.query('all')] == [
        13, 24, 27, 27, 27, 40, 35, 36, 13, 8, 7, 9
    ]
    keys = sorted(pp.cache.keys())
    assert np.all(pp.cache[keys[0]].conn_resids(0) == [0])
    assert np.all(pp.cache[keys[0]].conn_resids(1) == [12])
    assert np.all(pp.cache[keys[1]].conn_resids(0) == [0])
    assert np.all(pp.cache[keys[1]].conn_resids(1) == [11])
    assert np.all(pp.cache[keys[2]].conn_resids(0) == [0])
    assert np.all(pp.cache[keys[2]].conn_resids(1) == [8])
    assert np.all(pp.cache[keys[3]].conn_resids(0) == [0])
    assert np.all(pp.cache[keys[3]].conn_resids(1) == [6, 7, 8])
    assert np.all(pp.cache[keys[3]].conn_resids(2) == [9])
    assert np.all(pp.cache[keys[3]].conn_resids(3) == [16, 17])
    assert np.all(pp.cache[keys[3]].conn_resids(4) == [18, 19, 20])
    assert np.all(pp.cache[keys[3]].conn_resids(5) == [26])
    assert np.all(pp.cache[keys[4]].conn_resids(0) == [0])
    assert np.all(pp.cache[keys[4]].conn_resids(1) == [8])
    assert np.all(pp.cache[keys[4]].conn_resids(2) == [9])
    assert np.all(pp.cache[keys[4]].conn_resids(3) == [17])
    assert np.all(pp.cache[keys[4]].conn_resids(4) == [18])
    assert np.all(pp.cache[keys[4]].conn_resids(5) == [26])
    assert np.all(pp.cache[keys[5]].conn_resids(0) == [0])
    assert np.all(pp.cache[keys[5]].conn_resids(1) == [9])
    assert np.all(pp.cache[keys[6]].conn_resids(0) == [0])
    assert np.all(pp.cache[keys[6]].conn_resids(1) == [6])
    assert np.all(pp.cache[keys[7]].conn_resids(0) == [0])
    assert np.all(pp.cache[keys[7]].conn_resids(1) == [5])
    assert np.all(pp.cache[keys[8]].conn_resids(0) == [0])
    assert np.all(pp.cache[keys[8]].conn_resids(1) == [6, 7, 8, 9, 10, 11, 12])
    assert np.all(pp.cache[keys[9]].conn_resids(0) == [0])
    assert np.all(pp.cache[keys[9]].conn_resids(1) == [7])
    assert np.all(pp.cache[keys[10]].conn_resids(0) == [0])
    assert np.all(pp.cache[keys[10]].conn_resids(1) == [6])
    assert np.all(pp.cache[keys[11]].conn_resids(0) == [0])
    assert np.all(pp.cache[keys[11]].conn_resids(1) == [8])
