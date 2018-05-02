from glob import glob
import pyrosetta as pr
import sys
import os
sys.path.insert(0, '.')
print(os.getcwd())
from worms import util

pr.init()

template = """
{"file": "__DATADIR__/%s",
    "name": "%s" ,
    "class": ["%s"],
    "type": "?" ,
    "base": "?" ,
    "components": ["?"],
    "validated": false,
    "protocol": "?",
    "connections": %s
},
""".lstrip()

pdbs = ('loop.pdb', 'c3_splay.pdb', 'c6.pdb', 'c1.pdb', 'strand.pdb',
        'small.pdb', 'curved_helix.pdb', 'c3het.pdb', 'c4.pdb', 'c3.pdb',
        'c2.pdb', 'c5.pdb')

for pdb in pdbs:
    name = pdb.replace('.pdb', '')
    p = pr.pose_from_pdb('worms/data/' + pdb)
    cb = util.get_chain_bounds(p)
    connections = []
    for i, b in enumerate(cb):
        connections.append(
            dict(chain=i + 1, residues='%i,:1' % (i + 1), direction='N'))
        connections.append(
            dict(chain=i + 1, residues='%i,-1:' % (i + 1), direction='C'))
        if name in 'c1 c2 c3 c4 c5 c6':
            break
    print(template % (pdb, name, name, str(connections)))
