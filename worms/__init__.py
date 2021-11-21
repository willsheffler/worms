# -*- coding: utf-8 -*-
"""Top-level package for worms.
"""

__author__ = """Will Sheffler"""
__email__ = "willsheffler@gmail.com"
__version__ = "0.2.0"

import sys, os

#if 'pytest' in sys.modules:
#   os.environ['NUMBA_DISABLE_JIT'] = '1'

from worms.util.bunch import Bunch
from worms.util.ping import PING
from worms.util.timer import Timer
from worms.data import load

from deferred_import import deferred_import

app = deferred_import('worms.app')
bblock = deferred_import('worms.bblock')
cli = deferred_import('worms.cli')
criteria = deferred_import('worms.criteria')
data = deferred_import('worms.data')
database = deferred_import('worms.database')
edge = deferred_import('worms.edge')
edge_batch = deferred_import('worms.edge_batch')
filters = deferred_import('worms.filters')
output = deferred_import('worms.output')
rosetta_init = deferred_import('worms.rosetta_init')
search = deferred_import('worms.search')
ssdag = deferred_import('worms.ssdag')
ssdag_pose = deferred_import('worms.ssdag_pose')
tests = deferred_import('worms.tests')
topology = deferred_import('worms.topology')
util = deferred_import('worms.util')
vertex = deferred_import('worms.vertex')

# from worms.khash import * # khash_cffi needs updating for numba 0.49+
# from worms.bblock import *
# from worms.vertex import *
