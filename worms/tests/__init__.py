# -*- coding: utf-8 -*-
"""Unit test package for worms."""

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
    only_if_pyrosetta = pytest.mark.skip
