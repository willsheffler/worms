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
    pprint(test_db_files)
    pp = PDBPile(bakerdb_files=test_db_files, metaonly=True)
    assert len(pp.find_by_class('C4_C')) == 20
    assert len(pp.find_by_class('Het_C3_N')) == 21
    assert len(pp.find_by_class('Het_C2_N')) == 1
    assert len(pp.find_by_class('C5_N')) == 20
    assert len(pp.find_by_class('C6_N')) == 11
    assert len(pp.find_by_class('Het')) == 19494
    assert len(pp.find_by_class('C3_C')) == 56
    assert len(pp.find_by_class('Het_C2_C')) == 5
    assert len(pp.find_by_class('C3_N')) == 213
    assert len(pp.find_by_class('C2_C')) == 30
    assert len(pp.find_by_class('Het_C3_C')) == 30
    assert len(pp.find_by_class('C4_N')) == 9
    assert len(pp.find_by_class('C2_N')) == 11
    assert len(pp.find_by_class('Het:CCN')) == 1169
    assert len(pp.find_by_class('Het:NNC')) == 5962
    assert len(pp.find_by_class('Het:NNN')) == 1539
