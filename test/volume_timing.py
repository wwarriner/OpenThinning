import logging
import time

logging.basicConfig(level=logging.INFO, filemode="w")

from src.volume import *

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


@with_log()
def test_process_candidates__timing():
    SHAPE = 500

    lut = LookupTable.true()
    v = Volume.random((SHAPE, SHAPE, SHAPE))
    rng = np.random.default_rng()
    COUNTS = 10 ** np.arange(6)
    for COUNT in COUNTS:
        candidates = rng.integers(1, SHAPE, (COUNT, 1, 3), dtype=np.intp)

        start = time.perf_counter_ns()
        _ = v._process_candidates(lut, candidates)
        stop = time.perf_counter_ns()
        dur = stop - start

        logging.info(f"{dur}, {dur / COUNT}")


@with_log()
def test_restrict_candidates__timing():
    SHAPE = 500

    lut = LookupTable.true()
    v = Volume.random((SHAPE, SHAPE, SHAPE))
    rng = np.random.default_rng()
    COUNTS = 10 ** np.arange(6)
    for COUNT in COUNTS:
        candidates = rng.integers(1, SHAPE, (COUNT, 1, 3), dtype=np.intp)

        start = time.perf_counter_ns()
        _ = v._restrict_candidates(lut, candidates)
        stop = time.perf_counter_ns()
        dur = stop - start

        logging.info(f"{dur}, {dur / COUNT}")


@with_log()
def test_get_candidates__timing():
    SHAPE = 300

    for _ in range(10):
        v = Volume.random((SHAPE, SHAPE, SHAPE))

        start = time.perf_counter_ns()
        candidates = v._get_candidates(Axis.X, Offset.N)
        stop = time.perf_counter_ns()
        dur = stop - start

        logging.info(f"{dur}, {candidates.shape[0]}")


# @with_log(format="%(message)s")
def test_thin__timing():
    SHAPE = (26, 26, 26)
    RADIUS = max(s // 2 for s in SHAPE)

    lut = LookupTable.thinning_medial_surface()
    v = Volume.ball(SHAPE, RADIUS, interior=True)
    v._write_vtk("ball.vtk")
    v.apply_lut(lut, PurePath("ball-perf.log"))
    v._write_vtk("ball-skel.vtk")

    v = Volume.ball(SHAPE, RADIUS, interior=False)
    v._write_vtk("antiball.vtk")
    v.apply_lut(lut, PurePath("anti-ball-perf.log"))
    v._write_vtk("antiball-skel.vtk")


if __name__ == "__main__":
    test_thin__timing()
