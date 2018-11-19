from worms.criteria.hash_util import *
import numpy as np


def test_encode_decode_indices():
    for N in range(1, 5):
        for i in range(100):
            sizes = np.random.randint(4000, size=N) + 2
            idxes = np.array([np.random.randint(size) for size in sizes])
            index = encode_indices(sizes, idxes)
            idxs2 = decode_indices(sizes, index)
            assert np.all(idxes == idxs2)
