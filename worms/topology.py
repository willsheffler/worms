from copy import deepcopy
from collections import defaultdict
import itertools as it


def check_cycles(d, k, seenit):
    if k in seenit: return False
    seenit.add(k)
    if k in d:
        for k2 in d[k]:
            if not check_cycles(d, k2, seenit):
                return False
    return True


def enum_paths(d, path0):
    if path0[-1] not in d: return [path0]
    paths = list()
    for k in d[path0[-1]]:
        paths.extend(enum_paths(d, path0 + [k]))
    return paths


class Topology:
    def __init__(self, topolist):
        if topolist == [-1]: topolist = []
        self.topolist = topolist
        if topolist != [-1]:
            self.forward = defaultdict(list)
            self.backward = defaultdict(list)
            for a, b in zip(topolist[::2], topolist[1::2]):
                self.forward[a].append(b)
                self.backward[b].append(a)
        for k in self.forward:
            assert check_cycles(self.forward, k, set())
        for k in self.backward:
            assert check_cycles(self.backward, k, set())

    def paths(self):
        return enum_paths(self.forward, [0])

    def common_prefix(self):
        paths = self.paths()
        minlen = min(len(p) for p in paths)
        for i in range(minlen):
            if 1 < len(set(p[i] for p in paths)):
                return i
        return minlen

    def check_nc(self, nc, vals='NC', nullval='_'):
        if len(self.forward) is 0:
            return self.check_nc_linear(nc)

        for i in range(len(nc)):
            assert i in self.forward or i in self.backward
            if i not in self.forward:
                assert nc[i][1] == nullval
            if i not in self.backward:
                assert nc[i][0] == nullval

        for a, bs in self.forward.items():
            assert nc[a][1] in vals
            for b in bs:
                assert nc[b][0] in vals
                assert nc[a][1] != nc[b][0]

        for a, bs in self.backward.items():
            assert nc[a][0] in vals
            for b in bs:
                assert nc[b][1] in vals
                assert nc[a][0] != nc[b][1]

    def check_nc_linear(self, nc):
        assert len(
            nc[0]
        ) is 2, 'all connections should have two characters N C or _'
        assert nc[0][0] is '_', 'first connection should begin with _'
        assert nc[-1][1] is '_', 'last connection should end with _'
        for i in range(1, len(nc)):
            assert len(nc[i]) is 2
            prev = nc[i - 1][1]
            curr = nc[i][0]
            check = ((prev is 'N' and curr is 'C')
                     or (prev is 'C' and curr is 'N'))
            assert check, 'all connections must go CN or NC'