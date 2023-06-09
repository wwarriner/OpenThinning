import itertools
import logging
import time
import unittest

import numpy as np
import numpy.testing as npt

from src.lookup_table import *
from src.volume import *

logging.basicConfig(filename="timing.log", level=logging.INFO, filemode="w")


def cartesian(arrays, out=None):
    """
    Generate a Cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the Cartesian product of.
    out : ndarray
        Array to place the Cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing Cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    # m = n / arrays[0].size
    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            # for j in xrange(1, arrays[0].size):
            out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]
    return out


class LookupTableTest(unittest.TestCase):
    def test_symmetry(self):
        lut = LookupTable.read(PurePath("Data/LookupTables/Thinning_MedialSurface.bin"))
        tested = LookupTable.false()

        exhaustive = cartesian(itertools.repeat([False, True], 26))

        for item in exhaustive:
            if item[12] != item[14]:
                v1 = lut[item]
                reflect_x(item)
                v2 = lut[item]
                # self.assertEqual(v1, v2)
                if v1 != v2:
                    c = item.copy()
                    reflect_x(c)
                    logging.info(f"{packbits(c)}, {packbits(item)}")

            if item[10] != item[16]:
                v1 = lut[item]
                reflect_y(item)
                v2 = lut[item]
                # self.assertEqual(v1, v2)
                if v1 != v2:
                    c = item.copy()
                    reflect_y(c)
                    logging.info(f"{packbits(c)}, {packbits(item)}")

            if item[4] != item[22]:
                v1 = lut[item]
                reflect_z(item)
                v2 = lut[item]
                # self.assertEqual(v1, v2)
                if v1 != v2:
                    c = item.copy()
                    reflect_z(c)
                    logging.info(f"{packbits(c)}, {packbits(item)}")

    def test_getitem_exhaustive(self):
        # maybe this is foolish? it is 1.625 gib of data
        # need a fast way to generate it
        lut = LookupTable.random()
        exhaustive = cartesian(itertools.repeat([False, True], 26))
        npt.assert_array_equal(lut[exhaustive], lut.raw)

    def test_getitem(self):
        lut = LookupTable.random()
        rng = np.random.default_rng()
        COUNTS = 10 ** np.arange(7)
        for COUNT in COUNTS:
            indices = rng.choice([False, True], (COUNT, 26)).astype(np.bool_)

            start = time.perf_counter_ns()
            _ = lut[indices]
            stop = time.perf_counter_ns()
            dur = stop - start

            logging.info(f"{dur}, {dur / COUNT}")

    def test_read(self):
        start = time.perf_counter_ns()
        _ = LookupTable.read(PurePath("Data/LookupTables/Thinning_MedialSurface.bin"))
        stop = time.perf_counter_ns()
        dur = stop - start

        logging.info(f"{dur}")
