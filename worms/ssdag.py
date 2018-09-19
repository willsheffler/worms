import sys
from time import time
from collections import Counter
import concurrent.futures as cf
import numpy as np
from worms import Vertex, Edge, precompute_splicedb
from worms.bblock import bblock_dump_pdb, _BBlock
from worms.vertex import _Vertex
from worms.edge import _Edge
import worms

from worms.util import InProcessExecutor
from pprint import pprint
from logging import info
import string


def _validate_bbs_verts(bbs, verts):
    assert len(bbs) == len(verts)
    for bb, vert in zip(bbs, verts):
        if vert is None: continue
        assert 0 <= np.min(vert.ibblock)
        assert np.max(vert.ibblock) < len(bb)


class SearchSpaceDag:
    def __init__(self, bbspec, bbs, verts, edges):
        _validate_bbs_verts(bbs, verts)
        assert isinstance(bbs[0][0], _BBlock)
        assert isinstance(verts[0], (_Vertex, type(None)))
        assert len(edges) == 0 or isinstance(edges[0], _Edge)
        if bbspec:
            assert len(bbspec) == len(bbs)
        assert len(edges) == 0 or len(edges) + 1 == len(verts)
        self.bbspec = bbspec
        self.bbs = tuple(bbs)
        self.verts = tuple(verts)
        self.edges = tuple(edges)

    def __getstate__(self):
        return (
            self.bbspec,
            [[x._state for x in bb] for bb in self.bbs],
            [x._state for x in self.verts],
            [x._state for x in self.edges]
        )

    def __setstate__(self, state):
        self.bbspec = state[0]
        self.bbs = tuple(tuple(_BBlock(*x) for x in bb) for bb in state[1])
        self.verts = tuple(_Vertex(*x) for x in state[2])
        self.edges = tuple(_Edge(*x) for x in state[3])
        _validate_bbs_verts(self.bbs, self.verts)
        assert len(self.bbs) == len(self.verts) == len(self.edges) + 1

    def get_bases(self, idx):
        assert len(idx) == len(self.verts)
        bases = list()
        for i in range(len(idx)):
            bb = self.bbs[i][self.verts[i].ibblock[idx[i]]]
            bases.append(bytes(bb.base).decode('utf-8'))
        return bases

    def get_base_hashes(self, idx):
        assert len(idx) == len(self.verts)
        bases = list()
        for i in range(len(idx)):
            bb = self.bbs[i][self.verts[i].ibblock[idx[i]]]
            bases.append(bb.basehash)
        return bases


def simple_search_dag(
        criteria,
        db=None,
        nbblocks=100,
        min_seg_len=15,
        parallel=False,
        verbosity=0,
        timing=0,
        modbbs=None,
        make_edges=True,
        merge_bblock=None,
        precache_splices=False,
        precache_only=False,
        bbs=None,
        only_seg=None,
        source=None,
        print_edge_summary=False,
        no_duplicate_bases=False,
        shuffle_bblocks=False,
        **kw
):
    bbdb, spdb = db
    queries, directions = zip(*criteria.bbspec)
    tdb = time()
    if bbs is None:
        bbs = list()
        bases = list()
        # exclude_bases = set()
        for iquery, query in enumerate(queries):
            msegs = [
                i + len(queries) if i < 0 else i
                for i in criteria.which_mergeseg()
            ]
            if iquery in msegs[1:]:
                print('seg', iquery, 'repeating bblocks from', msegs[0])
                bbs.append(bbs[msegs[0]])
                bases.append(bases[msegs[0]])
                continue
            bbs0 = bbdb.query(
                query,
                max_bblocks=nbblocks,
                shuffle_bblocks=shuffle_bblocks,
                parallel=parallel,
            )
            bbs.append(bbs0)
            bases.append(Counter(bytes(b.base).decode('utf-8') for b in bbs0))

            # too few unique bases to filter here
            # if False:
            # new_bases = [bytes(b.base).decode('utf-8') for b in bbs[-1]]
            # exclude_bases.update(new_bases)
            # print('N exclude_bases', len(exclude_bases))
        assert len(bbs) == len(queries)
        for i, v in enumerate(bbs):
            assert len(v) > 0, 'no bblocks for query: "' + queries[i] + '"'
        print('bblock queries:', str(queries))
        print('bblock numbers:', [len(b) for b in bbs])
        print('bblocks id:', [id(b) for b in bbs])
        print('bblock0 id ', [id(b[0]) for b in bbs])
        print('base_counts:')
        for query, basecount in zip(queries, bases):
            counts = ' '.join(f'{k}: {c}' for k, c in basecount.items())
            print(f'   {query:10}', counts)

        if criteria.is_cyclic:
            assert bbs[criteria.from_seg] is bbs[criteria.to_seg]

    else:
        bbs = bbs.copy()

    assert len(bbs) == len(criteria.bbspec)
    if modbbs: modbbs(bbs)
    if merge_bblock is not None and merge_bblock >= 0:
        # print('which_mergeseg', criteria.bbspec, criteria.which_mergeseg())
        for i in criteria.which_mergeseg():
            bbs[i] = (bbs[i][merge_bblock], )

    tdb = time() - tdb
    # info(
    # f'bblock creation time {tdb:7.3f} num bbs: ' +
    # str([len(x) for x in bbs])
    # )

    if precache_splices:
        bbnames = [[bytes(bb.file) for bb in bbtup] for bbtup in bbs]
        bbpairs = set()
        # for bb1, bb2, dirn1 in zip(bbnames, bbnames[1:], directions):
        for i in range(len(bbnames) - 1):
            bb1 = bbnames[i]
            bb2 = bbnames[i + 1]
            dirn1 = directions[i]
            rev = dirn1[1] == 'N'
            if bbs[i] is bbs[i + 1]:
                bbpairs.update((a, a) for a in bb1)
            else:
                bbpairs.update((b, a) if rev else (a, b)
                               for a in bb1
                               for b in bb2)
        precompute_splicedb(
            db, bbpairs, verbosity=verbosity, parallel=parallel, **kw
        )
    if precache_only:
        return bbs

    verts = [None] * len(queries)
    edges = [None] * len(queries[1:])
    if source:
        srcdirn = [''.join('NC_' [d] for d in source.verts[i].dirn)
                   for i in range(len(source.verts))] # yapf: disable
        srcverts, srcedges = list(), list()
        for i, bb in enumerate(bbs):
            for isrc, bbsrc in enumerate(source.bbs):
                if directions[i] != srcdirn[isrc]: continue
                if [b.filehash for b in bb] == [b.filehash for b in bbsrc]:
                    verts[i] = source.verts[isrc]
                    srcverts.append(isrc)
        for i, bb in enumerate(zip(bbs, bbs[1:])):
            bb0, bb1 = bb
            for isrc, bbsrc in enumerate(zip(source.bbs, source.bbs[1:])):
                bbsrc0, bbsrc1 = bbsrc
                if directions[i] != srcdirn[isrc]: continue
                if directions[i + 1] != srcdirn[isrc + 1]: continue
                he = [b.filehash for b in bb0] == [b.filehash for b in bbsrc0]
                he &= [b.filehash for b in bb1] == [b.filehash for b in bbsrc1]
                if not he: continue
                edges[i] = source.edges[isrc]
                srcedges.append(isrc)

        # print(
        #     'src', sum(x is not None for x in verts), 'of', len(verts),
        #     'verts', '\n   ', sum(x is not None for x in edges), 'of',
        #     len(edges), 'edges', '\n   isrc verts', srcverts,
        #     '\n   isrc edges', srcedges, '\n   lenbbs', len(bbs)
        # )
        # sys.stdout.flush()

    tvertex = time()
    exe = InProcessExecutor()

    if parallel:
        exe = cf.ThreadPoolExecutor(max_workers=parallel)
    with exe as pool:
        if only_seg is not None:
            save = bbs, directions
            bbs = [bbs[only_seg]]
            directions = [directions[only_seg]]
            verts = [verts[only_seg]]
        futures = list()
        for i, bb in enumerate(bbs):
            dirn = directions[i]
            if verts[i] is None:
                futures.append(
                    pool.submit(Vertex, bb, dirn, min_seg_len=min_seg_len)
                )
        verts_new = [f.result() for f in futures]
        isnone = [i for i in range(len(verts)) if verts[i] is None]
        for i, inone in enumerate(isnone):
            verts[inone] = verts_new[i]
        # print(i, len(verts_new), len(verts))
        if isnone:
            assert i + 1 == len(verts_new)
        assert all(v for v in verts)
        if only_seg is not None:
            verts = ([None] * only_seg + verts +
                     [None] * (len(queries) - only_seg - 1))
            bbs, directions = save
    tvertex = time() - tvertex
    # info(
    # f'vertex creation time {tvertex:7.3f} num verts ' +
    # str([v.len if v else 0 for v in verts])
    # )

    if make_edges:
        tedge = time()
        for i, e in enumerate(edges):
            if e is None:
                edges[i] = Edge(
                    verts[i],
                    bbs[i],
                    verts[i + 1],
                    bbs[i + 1],
                    splicedb=spdb,
                    verbosity=verbosity,
                    precache_splices=precache_splices,
                    **kw
                )
        tedge = time() - tedge
        if print_edge_summary:
            _print_edge_summary(edges)
        # info(
        # f'edge creation time {tedge:7.3f} num splices ' +
        # str([e.total_allowed_splices()
        # for e in edges]) + ' num exits ' + str([e.len for e in edges])
        # )
        spdb.sync_to_disk()

    toret = SearchSpaceDag(criteria.bbspec, bbs, verts, edges)
    if timing:
        toret = toret, tdb, tvertex, tedge
    return toret


def _print_edge_summary(edges):
    print('splice stats: ', end='')
    for e in edges:
        nsplices = e.total_allowed_splices()
        ntot = e.nout * e.nent
        print(f'({nsplices:,} {nsplices*100.0/ntot:5.2f}%)', end=' ')
    print()


def graph_dump_pdb(out, ssdag, idx, pos, join='splice', trim=True):
    close = False
    if isinstance(out, str):
        out = open(out, 'w')
        close = True
    assert len(idx) == len(pos)
    assert idx.ndim == 1
    assert pos.ndim == 3
    assert pos.shape[-2:] == (4, 4)
    chain, anum, rnum = 0, 1, 1
    for i, tup in enumerate(zip(ssdag.bbs, ssdag.verts, idx, pos)):
        bbs, vert, ivert, x = tup
        chain, anum, rnum = bblock_dump_pdb(
            out=out,
            bblock=bbs[vert.ibblock[ivert]],
            dirn=vert.dirn if trim else (2, 2),
            splice=vert.ires[ivert] if trim else (-1, -1),
            pos=x,
            chain=chain,
            anum=anum,
            rnum=rnum,
            join=join,
        )
    if close: out.close()