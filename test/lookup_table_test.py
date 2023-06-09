import itertools
import unittest

import numpy as np
import numpy.testing as npt

import src.iter_utils as iter_utils
from src.lookup_table import *
from src.volume import *

BOOLS = [False, True]


class LookupTableTest(unittest.TestCase):
    def test_getitem_exhaustive(self):
        lut = LookupTable.random()
        exhaustive = iter_utils.cartesian(itertools.repeat(BOOLS, 26))
        npt.assert_array_equal(lut[exhaustive], lut.raw)

    def test_getitem(self):
        lut = LookupTable.random()
        rng = np.random.default_rng()
        COUNTS = 10 ** np.arange(7)
        for COUNT in COUNTS:
            indices = rng.choice(BOOLS, (COUNT, 26)).astype(np.bool_)
            _ = lut[indices]

    def test_read(self):
        _ = LookupTable.thinning_medial_surface()
        _ = LookupTable.thinning_medial_axis()
        _ = LookupTable.thinning_simple()
