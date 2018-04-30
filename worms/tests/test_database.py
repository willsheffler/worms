import pytest
from worms.database import *
import logging
import json
from worms.util import InProcessExecutor

test_db_files = [
    '/home/rubul/database/fusion/hb_dhr/master_database_generation2.txt',
    '/home/yhsia/helixdock/database/HBRP_Cx_database.txt',
    '/home/rubul/database/fusion/hb_dhr/c6_database.txt',
    '/home/yhsia/helixfuse/rosetta_scripts_ver/processing/database/HFuse_Cx_database.20180219.txt',
    '/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-103_2.20180406.txt',
    '/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_2arm_database.Sh3e_3.20180406.txt',
    '/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_3arm_database.Sh13_3.20180406.txt',
    '/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_3arm_database.Sh29_3.20180406.txt',
    '/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_3arm_database.Sh3e_3.20180406.txt',
    '/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-112_2.20180406.txt',
    '/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-127_2.20180406.txt',
    '/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-13_2.20180406.txt',
    '/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-15_2.20180406.txt',
    '/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-34_2.20180406.txt',
    '/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-37_2.20180406.txt',
    '/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-39_2.20180406.txt',
    '/home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-9_2.20180406.txt',
    '/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_2arm_database.Sh13_3.20180406.txt',
    '/home/yhsia/helixfuse/asym_combine/processing/database/HFuse_het_3chain_2arm_database.Sh29_3.20180406.txt',
]


# make a real test...
@pytest.mark.skip
def test_database_simple(tmpdir, caplog):
    pp = PDBPile(bakerdb_files=test_db_files[:1], cachedir=tmpdir)
    assert 0
