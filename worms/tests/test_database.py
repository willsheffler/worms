import pytest
from worms.database import *
import logging
import json
from worms.util import InProcessExecutor
from pprint import pprint
from os.path import dirname

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
