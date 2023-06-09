import functools
import logging
import time
import unittest

import numpy as np
import numpy.testing as npt

from src.lookup_table import LookupTable
from src.volume import *

logging.basicConfig(level=logging.INFO, filemode="w")

FN_NAME = "__fn_name"


def with_log(base_name: str = "", *dargs, **dkwargs):
    def dec(fn):
        # @functools.wraps(fn)
        def _(*args, **kwargs):
            name = fn.__name__
            if base_name != "":
                name = "_".join([name, base_name])
            logging.basicConfig(
                filename=f"{name}.log",
                level=logging.INFO,
                filemode="w",
                force=True,
                *dargs,
                **dkwargs,
            )
            return fn(*args, **kwargs)

        return _

    return dec


def set_timing_file(name: str) -> None:
    logging.basicConfig(filename=f"{name}.log")


def cube_to_neighborhood(_v: np.ndarray) -> np.ndarray:
    arr = _v.ravel()
    mask = np.ones(np.prod(arr.shape), dtype=np.bool_)
    mask[13] = False
    arr = arr[mask]
    arr = np.expand_dims(arr, 0)
    return arr


class VolumeTest(unittest.TestCase):
    def test_get_neighborhood(self):
        center = np.full((1, 3), 2, dtype=np.intp)

        v = Volume.true((3, 3, 3))
        npt.assert_array_equal(
            v.get_neighborhood(center), np.ones((1, 26), dtype=np.bool_)
        )

        v = Volume.false((3, 3, 3))
        npt.assert_array_equal(
            v.get_neighborhood(center), np.zeros((1, 26), dtype=np.bool_)
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
            Volume(arr).get_neighborhood(center), cube_to_neighborhood(arr)
        )

        """
        Testing indexing order matches what is expected by lookup table.
        """
        arr = np.zeros((3, 3, 3), dtype=np.bool_)
        arr[0, 0, 0] = True
        npt.assert_array_equal(
            Volume(arr).get_neighborhood(center), cube_to_neighborhood(arr)
        )

    def test_get_candidates(self):
        arr = np.ones((1, 1, 2), dtype=np.bool_)
        v = Volume(arr)
        cn = v.get_candidates(Axes.Z, Offset.N)
        cp = v.get_candidates(Axes.Z, Offset.P)
        npt.assert_array_equal(cn, np.array([[[1, 1, 2]]], dtype=np.intp))
        npt.assert_array_equal(cp, np.array([[[1, 1, 1]]], dtype=np.intp))

    def test_process_candidates(self):
        pass

    @with_log()
    def test_process_candidates__timing(self):
        SHAPE = 500

        lut = LookupTable.true()
        v = Volume.random((SHAPE, SHAPE, SHAPE))
        rng = np.random.default_rng()
        COUNTS = 10 ** np.arange(6)
        for COUNT in COUNTS:
            candidates = rng.integers(1, SHAPE, (COUNT, 1, 3), dtype=np.intp)

            start = time.perf_counter_ns()
            _ = v.process_candidates(lut, candidates)
            stop = time.perf_counter_ns()
            dur = stop - start

            logging.info(f"{dur}, {dur / COUNT}")

    @with_log()
    def test_restrict_candidates__timing(self, **kwargs):
        SHAPE = 500

        lut = LookupTable.true()
        v = Volume.random((SHAPE, SHAPE, SHAPE))
        rng = np.random.default_rng()
        COUNTS = 10 ** np.arange(6)
        for COUNT in COUNTS:
            candidates = rng.integers(1, SHAPE, (COUNT, 1, 3), dtype=np.intp)

            start = time.perf_counter_ns()
            _ = v.restrict_candidates(lut, candidates)
            stop = time.perf_counter_ns()
            dur = stop - start

            logging.info(f"{dur}, {dur / COUNT}")

    @with_log()
    def test_get_candidates__timing(self):
        SHAPE = 300

        for _ in range(10):
            v = Volume.random((SHAPE, SHAPE, SHAPE))

            start = time.perf_counter_ns()
            candidates = v.get_candidates(Axes.X, Offset.N)
            stop = time.perf_counter_ns()
            dur = stop - start

            logging.info(f"{dur}, {candidates.shape[0]}")

    @with_log(format="%(message)s")
    def test_thin__timing(self):
        SHAPE = (26, 26, 26)
        RADIUS = max(s // 2 for s in SHAPE)

        lut = LookupTable.read(PurePath("Data/LookupTables/Thinning_MedialSurface.bin"))
        v = Volume.ball(SHAPE, RADIUS, interior=True)
        v.write_vtk("ball.vtk")
        v.thin(lut)
        v.write_vtk("ball-skel.vtk")

        v = Volume.ball(SHAPE, RADIUS, interior=False)
        v.write_vtk("antiball.vtk")
        v.thin(lut)
        v.write_vtk("antiball-skel.vtk")
