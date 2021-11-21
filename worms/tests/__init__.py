# -*- coding: utf-8 -*-
"""Unit test package for worms."""

import os, importlib, pytest

from worms.tests.common_test_utils import *

HAVE_PYROSETTA = False

pyros = importlib.machinery.PathFinder().find_spec('pyrosetta')
if pyros:
   only_if_pyrosetta = lambda _: _
   HAVE_PYROSETTA = True
   pyrosd = importlib.machinery.PathFinder().find_spec('pyrosetta.distributed')
   if pyrosd:
      only_if_pyrosetta_distributed = lambda _: _
   else:
      only_if_pyrosetta_distributed = pytest.mark.skip
else:
   only_if_pyrosetta = only_if_pyrosetta_distributed = pytest.mark.skip

only_if_jit = lambda x: x
if "NUMBA_DISABLE_JIT" in os.environ:
   only_if_jit = pytest.mark.skip
