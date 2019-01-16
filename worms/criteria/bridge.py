from copy import deepcopy

from .base import *

from xbin import numba_xbin_indexer
from worms.khash import KHashi8i8
from worms.khash.khash_cffi import _khash_set, _khash_get
from worms.criteria.hash_util import encode_indices, decode_indices
import homog as hg
from worms.search.result import ResultJIT, SearchStats


class HashCriteria(WormCriteria):
    def __init__(
            self,
            from_seg=0,
            to_seg=-1,
            hash_cart_resl=None,
            hash_ori_resl=None,
            filter_hash=None,
            filter_binner=None,
            prev_isite=None,
            prev_vsize=None,
            **kw
    ):
        self.from_seg = from_seg
        self.to_seg = to_seg
        self.filter_hash = filter_hash
        self.filter_binner = filter_binner
        self.hash_cart_resl = hash_cart_resl
        self.hash_ori_resl = hash_ori_resl
        if self.hash_cart_resl is not None:
            self.hash_table = KHashi8i8()
            self.binner = numba_xbin_indexer(hash_cart_resl, hash_ori_resl)
            self.produces_no_results = True
        else:
            self.hash_table = None
            self.binner = None
        self.prev_isite = prev_isite
        self.prev_vsize = prev_vsize
        self.is_cyclic = False

    def score(self):
        raise NotImplementedError

    def cloned_segments(self):
        return self.to_seg,

    def jit_lossfunc(self):
        from_seg = self.from_seg
        to_seg = self.to_seg
        prev_isite = self.prev_isite
        prev_vsize = self.prev_vsize
        if self.binner:
            binner = self.binner
            hash_vp = self.hash_table.hash
        else:
            binner = hash_vp = None
        if self.filter_binner:
            filter_binner = self.filter_binner
            filter_hash_vp = self.filter_hash.hash
        else:
            filter_binner = filter_hash_vp = None

        if binner and filter_binner:  # filter and bin

            @jit
            def func(pos, idx, verts):
                xhat = pos[to_seg] @ np.linalg.inv(pos[from_seg])
                xhat = xhat.astype(np.float64)
                filter_key = filter_binner(xhat)
                if _khash_get(filter_hash_vp, filter_key, -1) < 0:
                    return 9e9
                key = binner(xhat.astype(np.float64))

                nverts = np.zeros((len(verts), ), dtype=np.int64)
                for i in range(len(nverts)):
                    nverts[i] = len(verts[i].ibblock)
                index = encode_indices(nverts, idx)

                _khash_set(hash_vp, key, index)
                return 9e9

            return func

        elif binner:  # bin without filter

            @jit
            def func(pos, idx, verts):
                xhat = pos[to_seg] @ np.linalg.inv(pos[from_seg])
                xhat = xhat.astype(np.float64)
                key = binner(xhat.astype(np.float64))
                _khash_set(hash_vp, key, 1)
                return 9e9

            return func

        elif filter_binner:  # filter only
            assert prev_isite is not None
            assert prev_vsize is not None

            @jit
            def func(pos, idx, verts):
                xhat = pos[to_seg] @ np.linalg.inv(pos[from_seg])
                xhat = xhat.astype(np.float64)
                filter_key = filter_binner(xhat)
                val = _khash_get(filter_hash_vp, filter_key, -1)
                if val < 0: return 9e9
                prev_ivert = decode_indices(prev_vsize, val)[0]
                prev_site = prev_isite[prev_ivert, 1]
                site = verts[-1].isite[idx[-1], 0]
                if prev_site == site: return 9e9
                return 0

            return func


class Bridge(WormCriteria):
    """
    Bridge grows a cyclic system in an A->B->A fashion
    Because growing A->...->A cycle is inefficient, split into
    A->B and B->A then match via hashing. Hash table built this
    would be inconveniently large, so this is done in three stages:
    stage1: grow A->B and produce coarse_hash of all B positions
    stage2: grow B->A if B is in coarse_hash, put in fine_hash
    stage3: grow A->B if B is in fine_hash, record result
    """

    def __init__(self, from_seg=0, to_seg=-1, *, tolerance=1.0, lever=25):
        self.from_seg = from_seg
        self.to_seg = to_seg
        self.tolerance = tolerance
        self.lever = lever
        self.rot_tol = tolerance / lever
        self.is_cyclic = False
        self.origin_seg = None
        self.symname = None

    def cloned_segments(self):
        return (len(self.bbspec) - 1) // 2,

    def merge_segment(self, **kw):
        return (len(self.bbspec) - 1) // 2

    def score(self, pos):
        x_from = pos[self.from_seg]
        x_to = pos[self.to_seg]
        xhat = x_to @ np.linalg.inv(x_from)
        axis, angle = hm.axis_angle_of(xhat)
        rot_err_sq = angle**2 * self.lever**2
        cart_err_sq = np.sum(xhat[:3, 3]**2)
        return np.sqrt(rot_err_sq + cart_err_sq)

    def alignment(self, pos, **kw):
        return np.eye(4)

    def stages(
            self, bbs, hash_cart_resl, hash_ori_resl, loose_hash_cart_resl,
            loose_hash_ori_resl, **kw
    ):
        """
        stage1: grow A->B and produce coarse_hash of all B positions
        stage2: grow B->A if B is in coarse_hash, put in fine_hash
        stage3: grow A->B if B is in fine_hash, record result
        """

        critA = HashCriteria(
            from_seg=0,
            to_seg=-1,
            # hash_cart_resl=6,
            # hash_ori_resl=25,
            hash_cart_resl=loose_hash_cart_resl,
            hash_ori_resl=loose_hash_ori_resl,
            merge_seg=self.merge_segment()
        )
        assert len(self.bbspec) == len(bbs)

        mbb = kw['merge_bblock']
        mseg = self.merge_segment()
        critA.bbspec = deepcopy(self.bbspec[:mseg + 1])
        critA.bbspec[-1][1] = critA.bbspec[-1][1][0] + '_'
        bbspecB = deepcopy(self.bbspec[mseg:])
        bbspecB[0][1] = '_' + bbspecB[0][1][1]
        bbsA = bbs[:mseg + 1]
        bbsB = bbs[mseg:]

        # print('Brigde.stages')
        # print('   ', critA.bbspec)
        # print('   ', bbspecB)

        def critB(prevcrit, prevssdag, prevresult):
            print(
                f'mbb {mbb:4}a hash: {prevcrit.hash_table.size():8,},',
                f' ntotal: {prevresult.stats.total_samples[0]:10,}    '
            )

            critB = HashCriteria(
                from_seg=-1,  # note: backwards! coming from other end
                to_seg=0,  # in stage B, to_seg becomes origin here
                # from_seg=0,
                # to_seg=-1,
                filter_binner=prevcrit.binner,
                filter_hash=prevcrit.hash_table,
                hash_cart_resl=hash_cart_resl,
                hash_ori_resl=hash_ori_resl,
                **kw
            )
            critB.bbspec = bbspecB
            return critB

        def critC(prevcrit, prevssdag, prevresult):
            print(
                f'mbb {mbb:4}b hash: {prevcrit.hash_table.size():8,},',
                f'ntotal: {prevresult.stats.total_samples[0]:10,}    '
            )
            critC = HashCriteria(
                from_seg=0,
                to_seg=-1,
                filter_binner=prevcrit.binner,
                filter_hash=prevcrit.hash_table,
                prev_isite=prevssdag.verts[0].isite,
                prev_vsize=np.array([len(v.ibblock) for v in prevssdag.verts]),
                **kw
            )
            critC.bbspec = critA.bbspec
            return critC

        return [(critA, bbsA),
                (critB, bbsB),
                (critC, bbsA)], merge_results_bridge

    def iface_rms(self, pose, prov, **kw):
        return -1


def merge_results_bridge(criteria, critC, ssdag, ssdB, ssdC, rsltC, **kw):

    # look up rsltCs in critC hashtable to get Bs

    sizesB = np.array([len(v.ibblock) for v in ssdB.verts])
    # print('merge_results_bridge')
    # print('    sizesB:', sizesB)
    # print('    sizes', [len(v.ibblock) for v in ssdag.verts])

    idx_list = list()
    pos_list = list()
    err_list = list()

    for iresult in range(len(rsltC.err)):
        idxC = rsltC.idx[iresult]
        posC = rsltC.pos[iresult]
        xhat = posC[criteria.to_seg] @ np.linalg.inv(posC[criteria.from_seg])
        xhat = xhat.astype(np.float64)
        key = critC.filter_binner(xhat)
        val = critC.filter_hash.get(key)
        assert val < np.prod(sizesB)
        idxB = decode_indices(sizesB, val)

        merge_ibblock = ssdB.verts[0].ibblock[idxB[0]]
        merge_ibblock_c = ssdC.verts[-1].ibblock[idxC[-1]]
        if merge_ibblock != merge_ibblock_c: continue
        merge_site1 = ssdB.verts[0].isite[idxB[0], 1]
        merge_site2 = ssdC.verts[-1].isite[idxC[-1], 0]
        # print('    merge_sites', iresult, merge_site1, merge_site2)
        if merge_site1 == merge_site2: continue

        merge_outres = ssdB.verts[0].ires[idxB[0], 1]
        merge_inres = ssdC.verts[-1].ires[idxC[-1], 0]

        iinC = [v.ires[i, 0] for v, i in zip(ssdC.verts, idxC)]
        iinB = [v.ires[i, 0] for v, i in zip(ssdB.verts, idxB)]
        ioutC = [v.ires[i, 1] for v, i in zip(ssdC.verts, idxC)]
        ioutB = [v.ires[i, 1] for v, i in zip(ssdB.verts, idxB)]

        ibbC = [v.ibblock[i] for v, i in zip(ssdC.verts, idxC)]
        ibbB = [v.ibblock[i] for v, i in zip(ssdB.verts, idxB)]
        # print('    hash stuff', iresult, key, val, idxB, ibbC, ibbB)
        # print('    iinC', iinC)
        # print('    iinB', iinB)
        # print('    ioutC', ioutC)
        # print('    ioutB', ioutB)
        # print('    ', merge_ibblock, merge_inres, merge_outres)

        imergeseg = len(idxC) - 1
        vmerge = ssdag.verts[imergeseg]
        w = ((vmerge.ibblock == merge_ibblock) *
             (vmerge.ires[:, 0] == merge_inres) *
             (vmerge.ires[:, 1] == merge_outres))
        imerge = np.where(w)[0]
        if len(imerge) is 0:
            print('    empty imerge')
            continue
        if len(imerge) > 1:
            print('    imerge', imerge)
            assert len(imerge) == 1
        imerge = imerge[0]
        # print('    ', imerge)
        # print('    ', vmerge.ibblock[imerge], vmerge.ires[imerge])
        idx = np.concatenate([idxC[:-1], [imerge], idxB[1:]])
        # print('    idx', idx)

        # compute pos and err
        spos = np.eye(4)
        pos = list()
        for i, v in enumerate(ssdag.verts):
            index = idx[i]
            pos.append(spos @ v.x2orig[index])
            spos = spos @ v.x2exit[index]
        pos = np.stack(pos)
        # print('posC')
        # for x in posC:
        # print(x)
        # print('pos')
        # for x in pos:
        # print(x)
        err = criteria.score(pos)
        # print('    err', err)
        if err > kw['tolerance']: continue

        idx_list.append(idx)
        pos_list.append(pos)
        err_list.append(err)

    if len(pos_list) > 0:
        return ResultJIT(
            np.stack(pos_list),
            np.stack(idx_list),
            np.array(err_list),
            SearchStats(0, 0, 0),
        )
    return None
