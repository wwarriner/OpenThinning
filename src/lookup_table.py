from pathlib import PurePath
from typing import Any

import numpy as np

BoolArrayT = np.ndarray[Any, np.dtype[np.bool_]]


class LookupTable:
    SIZE = 67_108_864  # 2 ** 26

    def __init__(self, _v: BoolArrayT, /) -> None:
        # bool(2**26)
        assert _v.dtype == np.bool_
        assert _v.ndim == 1
        assert _v.shape[0] == self.SIZE

        self._v: BoolArrayT = _v

    def __getitem__(self, _v: BoolArrayT, /) -> BoolArrayT:
        # bool(..., 26) -> bool(...)
        index = np.packbits(
            np.flip(_v, axis=-1),
            axis=-1,
            bitorder="little",
        ).view(
            np.uint32
        )  # bool(..., 26) -> uint32(...)
        # sum([x << i for i, x in enumerate(reversed(_v[0, ...].astype(np.uint32)))])
        entries = self._v.take(index[..., 0])  # uint32(...) -> bool(...)
        return entries

        # TODO return self.getitem(self._v, _v)

    @property
    def raw(self) -> BoolArrayT:
        return self._v

    @classmethod
    def read(cls, _path: PurePath, /) -> "LookupTable":
        packed_bytes = np.fromfile(_path, dtype=np.uint8)
        unpacked_bytes = np.unpackbits(packed_bytes)
        v = unpacked_bytes.astype(np.bool_)
        return cls(v)

    @classmethod
    def false(cls) -> "LookupTable":
        return cls(np.zeros((cls.SIZE,), dtype=np.bool_))

    @classmethod
    def true(cls) -> "LookupTable":
        return cls(np.ones((cls.SIZE,), dtype=np.bool_))

    @classmethod
    def random(cls, /) -> "LookupTable":
        rng = np.random.default_rng()
        return cls(rng.choice([False, True], size=(cls.SIZE,)))

    @classmethod
    def thinning_medial_surface(cls) -> "LookupTable":
        return cls.read(PurePath("res/lut-thinning-medial-surface.bin"))

    @classmethod
    def thinning_medial_axis(cls) -> "LookupTable":
        return cls.read(PurePath("res/lut-thinning-medial-axis.bin"))

    @classmethod
    def thinning_simple(cls) -> "LookupTable":
        return cls.read(PurePath("res/lut-thinning-simple.bin"))
