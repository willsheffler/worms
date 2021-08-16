from copy import deepcopy

from .base import *

# from xbin import numba_xbin_indexer
from worms.khash import KHashi8i8
from worms.khash.khash_cffi import _khash_set, _khash_get
from worms.criteria.hash_util import encode_indices, decode_indices
from worms import homog as hg
from worms.search.result import ResultJIT, SearchStats

class Fiber(WormCriteria):
   """
    Fiber grows a helical symmetry:

        A - B - C - A'
            \
             D - A"

    X' = Xform(A, A')
    X" = Xform(A, A")
    Fiber forms iff X'**M == X"**N for M/N small integers

    separately grow A->A' and A->A", then match via hashing.
    Store X**(1-M) in table, lookup X**(1-N) in table subsequently
    Hash table built this would be inconveniently large, so this is
    done in three stages see member function 'stages'

    merge segment will be A/A'/A" ... but B must be same! how to do
    this? probably have to build A-B 'tetramer' then search

    """

   pass
