# -*- coding: utf-8 -*-
"""Unit test package for worms."""

import os, imp, pytest

from worms.tests.common import *

HAVE_PYROSETTA = False

try:
   imp.find_module('pyrosetta')
   only_if_pyrosetta = lambda _: _
   HAVE_PYROSETTA = True
   try:
      imp.find_module('pyrosetta.distributed')
      only_if_pyrosetta_distributed = lambda _: _
   except ImportError:
      only_if_pyrosetta_distributed = pytest.mark.skip
except ImportError:
   only_if_pyrosetta = only_if_pyrosetta_distributed = pytest.mark.skip

only_if_jit = lambda x: x
if "NUMBA_DISABLE_JIT" in os.environ:
   only_if_jit = pytest.mark.skip
