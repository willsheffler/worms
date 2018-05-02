#!/usr/bin/env python

import sys
import os

file = file0 = sys.argv[1]
print(file)
if not os.path.basename(file).startswith("test_"):
    file = os.path.dirname(file) + "/tests/test_" + os.path.basename(file)
    if os.path.exists(file):
        cmd = 'pytest --duration=5 %s' % file
    else:
        cmd = 'python %s' % file0
else:
    cmd = 'pytest --duration=5 %s' % file
if not file.endswith('.py'):
    cmd = 'pytest --duration=5'

print('cwd:', os.getcwd())
print('cmd:', cmd)
print('----------- running  ------------')

sys.stdout.flush()
os.system(cmd)
