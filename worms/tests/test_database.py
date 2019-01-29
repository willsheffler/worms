import pytest
from worms.database import *
import logging
import json
from worms.util import InProcessExecutor
from pprint import pprint
from os.path import dirname
import numba as nb
from worms.tests import only_if_pyrosetta, only_if_pyrosetta_distributed

only_if_pyrosetta = pytest.mark.skipif("not HAVE_PYROSETTA")

test_db_files = [
    "c6_database.json",
    "HBRP_Cx_database.json",
    "HFuse_Cx_database.20180219.json",
    "HFuse_het_2chain_2arm_database.ZCON-103_2.20180406.json",
    "HFuse_het_2chain_2arm_database.ZCON-112_2.20180406.json",
    "HFuse_het_2chain_2arm_database.ZCON-127_2.20180406.json",
    "HFuse_het_2chain_2arm_database.ZCON-13_2.20180406.json",
    "HFuse_het_2chain_2arm_database.ZCON-15_2.20180406.json",
    "HFuse_het_2chain_2arm_database.ZCON-34_2.20180406.json",
    "HFuse_het_2chain_2arm_database.ZCON-37_2.20180406.json",
    "HFuse_het_2chain_2arm_database.ZCON-39_2.20180406.json",
    "HFuse_het_2chain_2arm_database.ZCON-9_2.20180406.json",
    "HFuse_het_3chain_2arm_database.Sh13_3.20180406.json",
    "HFuse_het_3chain_2arm_database.Sh13_3.20180416.json",
    "HFuse_het_3chain_2arm_database.Sh29_3.20180406.json",
    "HFuse_het_3chain_2arm_database.Sh29_3.20180416.json",
    "HFuse_het_3chain_2arm_database.Sh34_3.20180416.json",
    "HFuse_het_3chain_2arm_database.Sh3e_3.20180406.json",
    "HFuse_het_3chain_3arm_database.Sh13_3.20180406.json",
    "HFuse_het_3chain_3arm_database.Sh13_3.20180416.json",
    "HFuse_het_3chain_3arm_database.Sh29_3.20180406.json",
    "HFuse_het_3chain_3arm_database.Sh29_3.20180416.json",
    "HFuse_het_3chain_3arm_database.Sh34_3.20180416.json",
    "HFuse_het_3chain_3arm_database.Sh3e_3.20180406.json",
    "master_database_generation2.json",
]
test_db_files = [dirname(__file__) + "/../data/" + f for f in test_db_files]


@only_if_pyrosetta
def test_database_simple(tmpdir, caplog):
    pp = CachingBBlockDB(dbfiles=test_db_files)
    assert len(pp.query_names("C3_N")) == 213
    assert len(pp.query_names("Het_C3_C")) == 30
    assert len(pp.query_names("C2_N")) == 11
    assert len(pp.query_names("Het:NN")) == 9805
    assert len(pp.query_names("Het:NNX")) == 7501
    assert len(pp.query_names("Het:NNY")) == 2304
    assert 7501 + 2304 == 9805
    # pp.load_cached_coord_into_memory(pp.query_names('C3_N'))
    # assert len(pp.cache) == 213


@only_if_pyrosetta_distributed
def test_construct_database_from_pdbs(tmpdir, datadir):
    pp = CachingBBlockDB(
        cachedirs=str(tmpdir),
        dbfiles=[os.path.join(datadir, "test_db_file.json")],
        lazy=False,
        read_new_pdbs=True,
    )

    # print([len(pp.pose(k)) for k in pp.query_names('all')])
    assert [len(pp.pose(k)) for k in pp.query_names("all")] == [
        13,
        24,
        27,
        27,
        27,
        40,
        35,
        36,
        13,
        8,
        7,
        9,
    ]
    keys = sorted(pp._bblock_cache.keys(), key=lambda x: pp._key_to_pdbfile[x])
    assert np.all(pp.bblock(keys[0]).conn_resids(0) == [0])
    assert np.all(pp.bblock(keys[0]).conn_resids(1) == [12])
    assert np.all(pp.bblock(keys[1]).conn_resids(0) == [0])
    assert np.all(pp.bblock(keys[1]).conn_resids(1) == [11])
    assert np.all(pp.bblock(keys[2]).conn_resids(0) == [0])
    assert np.all(pp.bblock(keys[2]).conn_resids(1) == [8])
    assert np.all(pp.bblock(keys[3]).conn_resids(0) == [0])
    assert np.all(pp.bblock(keys[3]).conn_resids(1) == [6, 7, 8])
    assert np.all(pp.bblock(keys[3]).conn_resids(2) == [9])
    assert np.all(pp.bblock(keys[3]).conn_resids(3) == [16, 17])
    assert np.all(pp.bblock(keys[3]).conn_resids(4) == [18, 19, 20])
    assert np.all(pp.bblock(keys[3]).conn_resids(5) == [26])
    assert np.all(pp.bblock(keys[4]).conn_resids(0) == [0])
    assert np.all(pp.bblock(keys[4]).conn_resids(1) == [8])
    assert np.all(pp.bblock(keys[4]).conn_resids(2) == [9])
    assert np.all(pp.bblock(keys[4]).conn_resids(3) == [17])
    assert np.all(pp.bblock(keys[4]).conn_resids(4) == [18])
    assert np.all(pp.bblock(keys[4]).conn_resids(5) == [26])
    assert np.all(pp.bblock(keys[5]).conn_resids(0) == [0])
    assert np.all(pp.bblock(keys[5]).conn_resids(1) == [9])
    assert np.all(pp.bblock(keys[6]).conn_resids(0) == [0])
    assert np.all(pp.bblock(keys[6]).conn_resids(1) == [6])
    assert np.all(pp.bblock(keys[7]).conn_resids(0) == [0])
    assert np.all(pp.bblock(keys[7]).conn_resids(1) == [5])
    assert np.all(pp.bblock(keys[8]).conn_resids(0) == [0])
    assert np.all(pp.bblock(keys[8]).conn_resids(1) == [6, 7, 8, 9, 10, 11, 12])
    assert np.all(pp.bblock(keys[9]).conn_resids(0) == [0])
    assert np.all(pp.bblock(keys[9]).conn_resids(1) == [7])
    assert np.all(pp.bblock(keys[10]).conn_resids(0) == [0])
    assert np.all(pp.bblock(keys[10]).conn_resids(1) == [6])
    assert np.all(pp.bblock(keys[11]).conn_resids(0) == [0])
    assert np.all(pp.bblock(keys[11]).conn_resids(1) == [8])


def test_conftest_pdbfile(bbdb):
    assert len(bbdb.query("all")) == 12
