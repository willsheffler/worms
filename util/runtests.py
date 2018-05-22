"""
usage: python runtests.py @

this script exists for easy editor integration
"""

import sys
import os
import re


def dispatch(file, pytest_args='--duration=5'):
    dispatch = {
        'segments.py': ['tests/test_segments.py', 'tests/test_search.py'],
        'database.py': ['tests/test_database.py'],
        'vis.py': ['tests/test_edge.py'],
        'pose_contortions.py': [
            'tests/test_pose_contortions.py', 'tests/test_segments.py',
            'tests/test_search.py'
        ],
    }
    file = os.path.relpath(file)
    path, bname = os.path.split(file)
    print('dispatch', file)
    if (not file.endswith('.py') or not file.startswith('worms/')):
        return 'PYTHONPATH=. python ' + file
    if not os.path.basename(file).startswith("test_"):
        if bname in dispatch:
            return ('pytest {pytest_args} '.format(**vars()) + ' '.join(
                (os.path.join(path, n) for n in dispatch[bname])))
        else:
            testfile = re.sub('^worms', 'worms/tests', path) + '/test_' + bname
            print(testfile)
            if os.path.exists(testfile):
                return 'pytest {pytest_args} {testfile}'.format(**vars())
            else:
                return 'PYTHONPATH=. python ' + testfile
    return 'pytest {pytest_args} {file}'.format(**vars())


if len(sys.argv) is 1:
    cmd = 'pytest'
elif len(sys.argv) is 2:
    if sys.argv[1].endswith(__file__):
        # cmd = 'pytest DUMMY_DEBUGGING_runtests.py'
        # cmd = 'python util/runtests.py worms/criteria/unbounded.py'
        cmd = 'pytest'
    else:
        cmd = dispatch(sys.argv[1])
else:
    print('usage: runtests.py FILE')

print('call:', sys.argv)
print('cwd:', os.getcwd())
print('cmd:', cmd)
print('=' * 20, 'util/runtests.py running cmd in cwd', '=' * 23)
sys.stdout.flush()
# if cmd.startswith('pytest '):
# os.putenv('NUMBA_OPT', '1')
# os.putenv('NUMBA_DISABLE_JIT', '1')
os.system(cmd)
print('=' * 20, 'util/runtests.py done', '=' * 37)
