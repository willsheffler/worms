#!/usr/bin/env python

import sys
import os

file = sys.argv[1]
if not os.path.basename(file).startswith("test_"):
    file = os.path.dirname(file) + "/tests/test_" + os.path.basename(file)
    if not os.path.exists(file):
        file = ''
if not file.endswith('.py'):
    file = ''

os.system('pytest --duration=5 %s' % file)
