import unittest

import numpy as np
import numpy.testing as npt

from src.volume import *


class VolumeTest(unittest.TestCase):
    def test_get_neighborhood(self):
        center = np.full((1, 3), 2, dtype=np.intp)

        v = Volume.true((3, 3, 3))
        npt.assert_array_equal(
            v._get_neighborhood(center), np.ones((1, 26), dtype=np.bool_)
        )

        v = Volume.false((3, 3, 3))
        npt.assert_array_equal(
            v._get_neighborhood(center), np.zeros((1, 26), dtype=np.bool_)
        )

        """
        A perfectly asymmetric (3, 3, 3) boolean array is used to verify
        indexing is unchanged through a round-trip. Each face of the (3, 3, 3)
        array has a distinct pattern, so if the test passes, neighborhood
        indexing order is correct.
            - x0p: [0, 1, 0], [0, 0, 0], [1, 0, 0] # oblique separated pair
            - x0n: [0, 1, 0], [0, 0, 1], [0, 0, 1] # "jay" shape
            - x1p: [0, 0, 0], [1, 1, 1], [0, 0, 0] # row
            - x1n: [1, 0, 0], [0, 1, 0], [0, 0, 1] # diagonal
            - x2p: [0, 0, 0], [0, 1, 0], [1, 0 ,0] # partial diagonal
            - x2n: [0, 0, 0], [0, 1, 1], [0, 0, 1] # small "ell"
        """
        arr = np.array(
            [
                [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
                [[0, 1, 0], [0, 0, 0], [0, 1, 1]],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            ],
            dtype=np.bool_,
        )
        npt.assert_array_equal(
            Volume(arr)._get_neighborhood(center), Volume.cube_to_neighborhood(arr)
        )

        """
        Testing indexing order matches what is expected by lookup table.
        """
        arr = np.zeros((3, 3, 3), dtype=np.bool_)
        arr[0, 0, 0] = True
        npt.assert_array_equal(
            Volume(arr)._get_neighborhood(center), Volume.cube_to_neighborhood(arr)
        )

    def test_get_candidates(self):
        arr = np.ones((1, 1, 2), dtype=np.bool_)
        v = Volume(arr)
        cn = v._get_candidates(Axis.Z, Offset.N)
        cp = v._get_candidates(Axis.Z, Offset.P)
        npt.assert_array_equal(cn, np.array([[[1, 1, 2]]], dtype=np.intp))
        npt.assert_array_equal(cp, np.array([[[1, 1, 1]]], dtype=np.intp))
