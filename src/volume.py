import itertools
import logging
import time
from pathlib import Path, PurePath
from typing import Any, Tuple, TypeVar

import numba
import numpy as np

BoolArrayT = np.ndarray[Any, np.dtype[np.bool_]]
IntArrayT = np.ndarray[Any, np.dtype[np.intp]]

PathLike = Path | PurePath | str
Shape = Tuple[int, int, int]

_T = TypeVar("_T")

Vector3 = Tuple[_T, _T, _T]


logging.basicConfig(
    filename="thinning.log",
    level=logging.INFO,
    filemode="w",
    format="%(message)s",
    force=True,
)

import enum

from src.lookup_table import LookupTable


class Axes(enum.Enum):
    X = 0
    Y = 1
    Z = 2


class Offset(enum.Enum):
    N = -1
    P = 1


NEIGHBORHOOD_OFFSETS = np.array(
    [
        # X,  Y,  Z
        # Z = -1
        [-1, -1, -1],
        [+0, -1, -1],
        [+1, -1, -1],
        [-1, +0, -1],
        [+0, +0, -1],
        [+1, +0, -1],
        [-1, +1, -1],
        [+0, +1, -1],
        [+1, +1, -1],
        # Z = +0
        [-1, -1, +0],
        [+0, -1, +0],
        [+1, -1, +0],
        [-1, +0, +0],
        # [+0, +0, +0],
        [+1, +0, +0],
        [-1, +1, +0],
        [+0, +1, +0],
        [+1, +1, +0],
        # Z = +1
        [-1, -1, +1],
        [+0, -1, +1],
        [+1, -1, +1],
        [-1, +0, +1],
        [+0, +0, +1],
        [+1, +0, +1],
        [-1, +1, +1],
        [+0, +1, +1],
        [+1, +1, +1],
        # X,  Y,  Z
    ],
    dtype=np.intp,
)
NEIGHBORHOOD_OFFSETS = np.expand_dims(NEIGHBORHOOD_OFFSETS, axis=0)
NEIGHBORHOOD_OFFSETS = np.flip(NEIGHBORHOOD_OFFSETS, axis=2)  # order is really Z, Y, X
assert NEIGHBORHOOD_OFFSETS.shape == (1, 26, 3)


class Volume:
    dtype = np.bool_

    def __init__(self, _v: BoolArrayT, /) -> None:
        assert _v.dtype == self.dtype
        assert _v.ndim == 3

        v = self._pad(_v)

        self._v: BoolArrayT = v  # bool(z, y, x)

    @property
    def size(self) -> int:
        return np.prod(self.shape).item()

    @property
    def shape(self) -> Vector3[int]:
        return tuple(x - 2 for x in self._v.shape)

    def thin(self, _lut: LookupTable, /) -> None:
        # modifies self
        logging.info("size,iteration,axis,offset,true_count,true_frac,time_ns")
        iteration = 0
        modified: bool = True
        while modified:
            modified = False
            for axis, offset in itertools.product(Axes, Offset):
                start_ns = time.perf_counter_ns()

                candidates = self.get_candidates(axis, offset)
                candidates = self.restrict_candidates(_lut, candidates)
                # modified |= self.process_candidates(_lut, candidates)
                modified |= self.process_candidates_numba(_lut, candidates)

                stop_ns = time.perf_counter_ns()
                duration_ns = stop_ns - start_ns

                true_count = self._v.sum()
                true_frac = true_count / self._v.size

                logging.info(
                    f"{self._v.size:d},{iteration:d},{axis.value:d},{offset.value:+d},{true_count:d},{true_frac:.16f},{duration_ns:d}"
                )
            iteration += 1

    def get_candidates(self, _axis: Axes, _offset: Offset, /) -> IntArrayT:
        # bool(z, y, x) -> int(k, 3)
        o = abs(_offset.value)

        # [1, N]
        pos = [slice(None, None, None)] * 3
        pos[_axis.value] = slice(o, None, None)

        # [0, N-1]
        neg = [slice(None, None, None)] * 3
        neg[_axis.value] = slice(None, self._v.shape[_axis.value] - o, None)

        neg_offset = _offset.value < 0
        pos_offset = 0 < _offset.value
        if pos_offset:
            value = self._v[*pos]  # indices decreased by one in _axis
            predecessor = self._v[*neg]
        elif neg_offset:
            value = self._v[*neg]  # indices as expected
            predecessor = self._v[*pos]
        else:
            assert False
        c = list(np.where(value & ~predecessor))
        if pos_offset:
            c[_axis.value] += 1
        elif neg_offset:
            pass
        else:
            assert False

        c = np.stack(c, axis=-1)
        c = np.expand_dims(c, axis=1)
        return c

    def restrict_candidates(
        self, _lut: LookupTable, _candidates: IntArrayT, /
    ) -> IntArrayT:
        # int(k, 3) -> int(k, 3)
        neighborhoods = self.get_neighborhood(_candidates)
        keep = _lut[neighborhoods[:, 0, :]]
        _candidates = _candidates[keep, ...]
        return _candidates

    def process_candidates_numba(
        self, _lut: LookupTable, _candidates: IntArrayT, /
    ) -> bool:
        return process_candidates(_lut.raw, _candidates, self._v)

    def process_candidates(self, _lut: LookupTable, _candidates: IntArrayT, /) -> bool:
        # modifies _v
        # int(k, 3), bool(z, y, x) -> bool(z, y, x)
        modified: bool = False
        for candidate in _candidates:
            neighborhood = self.get_neighborhood(candidate)
            if _lut[neighborhood].item():
                self._v[candidate[..., 0], candidate[..., 1], candidate[..., 2]] = False
                modified = True
        return modified

    def get_neighborhood(self, _candidates: IntArrayT, /) -> BoolArrayT:
        # int(k, 1, 3) -> bool(k, 1, 26)
        candidates = np.expand_dims(_candidates, axis=1)
        neighborhood = (
            candidates + NEIGHBORHOOD_OFFSETS
        )  # (k, 1, 3) + (1, 26, 3) -> (k, 26, 3)
        x = neighborhood[..., 0]
        y = neighborhood[..., 1]
        z = neighborhood[..., 2]
        out = self._v[x, y, z]
        return out

    def write_vtk(
        self,
        _path: PathLike,
        /,
        title: str = "data",
        origin: Vector3[float] = (0.0, 0.0, 0.0),
        spacing: Vector3[float] = (1.0, 1.0, 1.0),
        *,
        comment: str = "",
    ) -> None:
        b = lambda x: bytes(x, "utf-8")
        v = lambda x: f"{x[0]} {x[1]} {x[2]}"
        size = lambda x: x[0] * x[1] * x[2]

        shape = self.shape

        with open(_path, "wb") as f:
            f.write(b(f"# vtk DataFile Version 2.0\n"))
            f.write(b(f"{comment}\n"))
            f.write(b(f"BINARY\n"))
            f.write(b(f"DATASET STRUCTURED_POINTS\n"))
            f.write(b(f"DIMENSIONS {v(shape):s}\n"))
            f.write(b(f"SPACING {v(spacing):s}\n"))
            f.write(b(f"ORIGIN {v(origin):s}\n"))
            f.write(b(f"POINT_DATA {size(shape):d}\n"))
            f.write(b(f"SCALARS {title:s} unsigned_char 1\n"))
            f.write(b(f"LOOKUP_TABLE default\n"))
            f.write(np.moveaxis(self._unpad(self._v), (0, 1, 2), (2, 1, 0)).tobytes())
        pass

    @classmethod
    def false(cls, _shape: Shape, /) -> "Volume":
        return cls(cls._full(_shape, False))

    @classmethod
    def true(cls, _shape: Shape, /) -> "Volume":
        return cls(cls._full(_shape, True))

    @classmethod
    def random(cls, _shape: Shape, /) -> "Volume":
        rng = np.random.default_rng()
        return cls(rng.choice([False, True], size=_shape))

    @classmethod
    def box_cross(cls, _shape: Shape, /) -> "Volume":
        v = cls._blank(_shape)
        s = lambda x: slice(_shape[x] // 6, _shape[x] // 4, 1)
        v[:, :, s(2)]
        v[:, s(1), :]
        v[s(0), :, :]
        return cls(v)

    @classmethod
    def ball(cls, _shape: Shape, _radius: float, /, interior: bool = True) -> "Volume":
        center = lambda x: _shape[x] * 0.5
        XC = center(0)
        YC = center(1)
        ZC = center(2)
        X, Y, Z = cls._grid(_shape)

        DX = X - XC
        DY = Y - YC
        DZ = Z - ZC
        D2 = DX * DX + DY * DY + DZ * DZ
        R2 = _radius * _radius
        SET_INTERIOR = D2 <= R2

        v = cls._blank(_shape)
        v[SET_INTERIOR] = interior
        v[~SET_INTERIOR] = not interior
        return cls(v)

    @staticmethod
    def _blank(_shape: Shape, /) -> BoolArrayT:
        return Volume._full(_shape, False)

    @staticmethod
    def _full(_shape: Shape, _v: bool, /) -> BoolArrayT:
        return np.full(_shape, _v, dtype=Volume.dtype)

    @staticmethod
    def _grid(_shape: Shape, /) -> Tuple[IntArrayT, IntArrayT, IntArrayT]:
        vec = lambda x: np.arange(_shape[x], dtype=np.intp)
        X, Y, Z = np.meshgrid(
            vec(0),
            vec(1),
            vec(2),
            copy=False,
            indexing="ij",
        )
        return X, Y, Z

    @staticmethod
    def _pad(_v: BoolArrayT) -> BoolArrayT:
        return np.pad(_v, pad_width=1, mode="constant", constant_values=False)

    @staticmethod
    def _unpad(_v: BoolArrayT) -> BoolArrayT:
        return _v[1:-1, 1:-1, 1:-1]


spec = [
    ("_lut", numba.bool_[:]),
    ("_candidates", numba.int64[:, :]),
    ("_v", numba.bool_[:, :, :]),
]


@numba.njit  # (spec=spec)
def process_candidates(
    _lut: BoolArrayT, _candidates: IntArrayT, _v: BoolArrayT, /
) -> bool:
    # modifies _v
    # bool(m), int(k, 3), bool(z, y, x) -> bool(z, y, x)
    modified: bool = False
    for i in range(_candidates.shape[0]):
        candidate = _candidates[i, 0, :]  # (3)
        neighborhood = get_neighborhood(candidate, _v)  # (26)
        if getitem(_lut, neighborhood):
            x = candidate[0]
            y = candidate[1]
            z = candidate[2]
            _v[x, y, z] = False
            modified = True
    return modified


spec = [
    ("_candidates", numba.int64[:, :]),
    ("_v", numba.bool_[:, :, :]),
]


@numba.njit  # (spec=spec)
def get_neighborhood(_candidate: IntArrayT, _v: BoolArrayT, /) -> BoolArrayT:
    # int(3) -> bool(26)
    candidate = np.expand_dims(_candidate, axis=0)  # (1, 3)
    neighborhood = candidate + NEIGHBORHOOD_OFFSETS[0, ...]  # (26, 3)
    x = neighborhood[..., 0]  # (26)
    y = neighborhood[..., 1]  # (26)
    z = neighborhood[..., 2]  # (26)
    out = np.zeros((26), dtype=np.bool_)
    for i in range(26):
        out[i] = _v[x[i], y[i], z[i]]
    return out


@numba.njit
def getitem(_self: BoolArrayT, _v: BoolArrayT) -> BoolArrayT:
    # bool(26) -> bool
    index = packbits(_v)  # bool(26) -> uint32
    entry = _self[index]  # uint32 -> bool
    return entry


@numba.njit
def setitem(_self: BoolArrayT, _v: BoolArrayT, entry: bool) -> None:
    index = packbits(_v)
    _self[index] = entry


@numba.njit
def swap(_v: BoolArrayT, i1: numba.intp, i2: numba.intp) -> None:
    _v[i1], _v[i2] = _v[i2], _v[i1]


# symmetries to test

# reflection if any element of opposing faces differ
# rotation if any element of same-oriented edges differ
# mark checked ones as we go to avoid duplication


@numba.njit
def reflect_x(_v) -> None:
    swap(_v, 0, 2)
    swap(_v, 3, 5)
    swap(_v, 6, 8)
    swap(_v, 9, 11)
    # y center offset -1
    swap(_v, 12, 13)
    # x center offset -1
    swap(_v, 14, 16)
    swap(_v, 17, 19)
    swap(_v, 20, 22)
    swap(_v, 23, 25)


@numba.njit
def reflect_y(_v) -> None:
    swap(_v, 0, 6)
    swap(_v, 1, 7)
    swap(_v, 2, 8)
    # y center offset -1
    swap(_v, 9, 14)
    swap(_v, 10, 15)
    swap(_v, 11, 16)
    # x center offset -1
    swap(_v, 17, 23)
    swap(_v, 18, 24)
    swap(_v, 19, 25)


@numba.njit
def reflect_z(_v) -> None:
    # y center offset -1
    swap(_v, 0, 17)
    swap(_v, 1, 18)
    swap(_v, 2, 19)
    swap(_v, 3, 20)
    swap(_v, 4, 21)
    swap(_v, 5, 22)
    swap(_v, 6, 23)
    swap(_v, 7, 24)
    swap(_v, 8, 25)


@numba.njit
def packbits(_v: BoolArrayT) -> numba.uint32:
    p: numba.uint32 = numba.uint32(0)
    for i in range(26):
        p = p | (_v[i] << i)
    return p
