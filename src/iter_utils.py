import itertools
import logging
import time
from pathlib import PurePath
from typing import Any, Dict, Iterable, List

import numpy as np


class PerformanceCounter:
    def __init__(
        self,
        _formatters: Dict[str, str],
        /,
        log_filepath: PurePath = PurePath("perf.log"),
    ) -> None:
        logging.basicConfig(
            filename=log_filepath,
            level=logging.INFO,
            filemode="w",
            format="%(message)s",
            force=True,
        )
        self._iteration: int = 0
        self._formatters: Dict[str, str] = _formatters
        self._start_ns: int = time.perf_counter_ns()

    def __enter__(self) -> "PerformanceCounter":
        v = list(self._formatters.keys())
        v.insert(0, "iteration")
        v.insert(1, "time_ns")
        self._log(v)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def start(self) -> None:
        self._start_ns = time.perf_counter_ns()

    def stop(self, _v: Dict[str, Any]) -> None:
        stop_ns = time.perf_counter_ns()

        parts: List[str] = []
        for k, v in _v.items():
            try:
                f = self._formatters[k]
                s = f.format(v)
            except:
                s = ""
            parts.append(s)

        parts.insert(0, f"{self._iteration:d}")

        duration_ns = stop_ns - self._start_ns
        parts.insert(1, f"{duration_ns:d}")
        self._log(parts)

        self._start_ns = 0
        self._iteration += 1

    def _log(self, _v: Iterable[str]) -> None:
        logging.info(",".join(_v))


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
