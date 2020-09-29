# -*- coding: utf-8 -*-
"""Unit test package for worms."""

import os
import pytest

try:
   import pyrosetta

   HAVE_PYROSETTA = True
   only_if_pyrosetta = lambda x: x
   try:
      import pyrosetta.distributed

      HAVE_PYROSETTA_DISTRIBUTED = True
      only_if_pyrosetta_distributed = lambda x: x
   except ImportError:
      HAVE_PYROSETTA_DISTRIBUTED = False
      only_if_pyrosetta_distributed = pytest.mark.skip
except ImportError:
   HAVE_PYROSETTA = HAVE_PYROSETTA_DISTRIBUTED = False
   only_if_pyrosetta = only_if_pyrosetta_distributed = pytest.mark.skip

only_if_jit = lambda x: x
if "NUMBA_DISABLE_JIT" in os.environ:
   only_if_jit = pytest.mark.skip
