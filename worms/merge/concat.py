import numpy as np

from worms.search.result import ResultJIT, subset_result
from worms import util
from worms.criteria.hash_util import _get_hash_val
from worms.khash.khash_cffi import _khash_get
from worms.khash import KHashi8i8


def merge_results_concat(
        criteria, ssdag, ssdagA, rsltA, critB, ssdagB, rsltB, merged_err_cut,
        max_merge, **kw
):
    bsfull = [x[0] for x in ssdag.bbspec]
    bspartA = [x[0] for x in ssdagA.bbspec]
    bspartB = [x[0] for x in ssdagB.bbspec]
    assert bsfull[-len(bspartA):] == bspartA
    assert bsfull[:len(bspartB)] == bspartB

    # print('merge_results_concat ssdag.bbspec', ssdag.bbspec)
    # print('merge_results_concat criteria.bbspec', criteria.bbspec)
    rsltB = subset_result(rsltB, slice(max_merge))

    binner = critB.binner
    hash_table = critB.hash_table
    from_seg = criteria.from_seg

    assert len(ssdagB.bbs[-1]) == len(ssdagA.bbs[0])
    assert len(ssdagB.bbs[-1]) == len(ssdag.bbs[from_seg])
    assert len(ssdagB.bbs[-1]) == 1, 'did you set merge_bblock?'
    assert ssdagB.bbs[-1][0].filehash == ssdagA.bbs[0][0].filehash
    assert ssdagB.bbs[-1][0].filehash == ssdag.bbs[from_seg][0].filehash
    for _ in range(from_seg):
        f = [bb.filehash for bb in ssdag.bbs[_]]
        assert f == [bb.filehash for bb in ssdagB.bbs[_]]
    for _ in range(len(ssdag.verts) - from_seg):
        f = [bb.filehash for bb in ssdag.bbs[from_seg + _]]
        assert f == [bb.filehash for bb in ssdagA.bbs[_]]

    n = len(rsltB.idx)
    nv = len(ssdag.verts)
    merged = ResultJIT(
        pos=np.empty((n, nv, 4, 4), dtype='f4'),
        idx=np.empty((n, nv), dtype='i4'),
        err=9e9 * np.ones((n, ), dtype='f8'),
        stats=np.empty(n, dtype='i4')
    )
    ok = np.ones(n, dtype=np.bool)
    for i_in_rslt in range(n):
        # print(rsltB.pos[i_in_rslt, -1])
        val = _get_hash_val(
            binner, hash_table, rsltB.pos[i_in_rslt, -1], criteria.nfold
        )
        # print(
        # 'merge_results_concat', i_in_rslt, val, np.right_shift(val, 32),
        # np.right_shift(val, 16) % 16,
        # np.right_shift(val, 8) % 8, val % 8
        # )
        if val < 0:
            print('val < 0')
            ok[i_in_rslt] = False
            continue
        i_ot_rslt = np.right_shift(val, 32)
        assert i_ot_rslt < len(rsltA.idx)

        # check score asap
        pos = np.concatenate((
            rsltB.pos[i_in_rslt, :-1],
            rsltB.pos[i_in_rslt, -1] @ rsltA.pos[i_ot_rslt, :]
        ))
        assert np.allclose(pos[from_seg], rsltB.pos[i_in_rslt, -1])
        err = criteria.score(pos.reshape(-1, 1, 4, 4))
        merged.err[i_in_rslt] = err
        # print('merge_results_concat', i_in_rslt, pos)
        # print('merge_results_concat', i_in_rslt, err)
        if err > merged_err_cut: continue

        i_outer = rsltA.idx[i_ot_rslt, 0]
        i_outer2 = rsltA.idx[i_ot_rslt, -1]
        i_inner = rsltB.idx[i_in_rslt, -1]
        v_inner = ssdagB.verts[-1]
        v_outer = ssdagA.verts[0]
        ibb = v_outer.ibblock[i_outer]
        assert ibb == 0
        ires_in = v_inner.ires[i_inner, 0]
        ires_out = v_outer.ires[i_outer, 1]
        isite_in = v_inner.isite[i_inner, 0]
        isite_out = v_outer.isite[i_outer, 1]
        isite_out2 = ssdagA.verts[-1].isite[i_outer2, 0]
        mrgv = ssdag.verts[from_seg]
        assert max(mrgv.ibblock) == 0
        assert max(ssdagA.verts[-1].ibblock) == 0

        imerge = util.binary_search_pair(mrgv.ires, (ires_in, ires_out))
        if imerge == -1:
            # if imerge < 0:
            ok[i_in_rslt] = False
            continue
        idx = np.concatenate(
            (rsltB.idx[i_in_rslt, :-1],
             [imerge], rsltA.idx[i_ot_rslt, 1:])
        )
        assert len(idx) == len(ssdag.verts)
        for ii, v in zip(idx, ssdag.verts):
            if v is not None: assert ii < v.len
        assert len(pos) == len(idx) == nv
        merged.pos[i_in_rslt] = pos
        merged.idx[i_in_rslt] = idx
        merged.stats[i_in_rslt] = i_ot_rslt
    # print(merged.err[:100])
    nbad = np.sum(1 - ok)
    if nbad: print('bad imerge', nbad, 'of', n)
    # print('bad score', np.sum(merged.err > merged_err_cut), 'of', n)
    ok[merged.err > merged_err_cut] = False
    ok = np.where(ok)[0][np.argsort(merged.err[ok])]
    merged = subset_result(merged, ok)
    return merged
