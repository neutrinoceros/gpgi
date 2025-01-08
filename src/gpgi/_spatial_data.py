import enum
import math
from dataclasses import dataclass
from itertools import chain
from typing import Any, Protocol, TypeVar, assert_never, final

import numpy as np
from numpy.typing import NDArray

from gpgi._typing import FieldMap, Name, Real


class Geometry(enum.StrEnum):
    CARTESIAN = enum.auto()
    POLAR = enum.auto()
    CYLINDRICAL = enum.auto()
    SPHERICAL = enum.auto()
    EQUATORIAL = enum.auto()


_AXES_LIMITS: dict[Name, tuple[float, float]] = {
    "x": (-float("inf"), float("inf")),
    "y": (-float("inf"), float("inf")),
    "z": (-float("inf"), float("inf")),
    "radius": (0, float("inf")),
    "azimuth": (0, 2 * np.pi),
    "colatitude": (0, np.pi),
    "latitude": (-np.pi / 2, np.pi / 2),
}


class SpatialData(Protocol):
    geometry: Geometry
    axes: tuple[Name, ...]
    coordinates: FieldMap
    fields: FieldMap


T_contra = TypeVar("T_contra", bound=SpatialData, contravariant=True)


class Validator(Protocol[T_contra]):
    @classmethod
    def check(cls, data: T_contra) -> None: ...


@final
class GeometryValidator:
    @classmethod
    def check(cls, data: SpatialData) -> None:
        match data.geometry:
            case Geometry.CARTESIAN:
                axes3D = ("x", "y", "z")
            case Geometry.POLAR:
                axes3D = ("radius", "azimuth", "z")
            case Geometry.CYLINDRICAL:
                axes3D = ("radius", "z", "azimuth")
            case Geometry.SPHERICAL:
                axes3D = ("radius", "colatitude", "azimuth")
            case Geometry.EQUATORIAL:
                axes3D = ("radius", "azimuth", "latitude")
            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

        for i, (expected, actual) in enumerate(zip(axes3D, data.axes, strict=False)):
            if actual != expected:
                raise ValueError(
                    f"Got invalid axis name {actual!r} on position {i}, "
                    f"with geometry {data.geometry.name.lower()!r}\n"
                    f"Expected axes ordered as {axes3D[: len(data.axes)]}"
                )


@final
class BasicCoordinatesValidator:
    @classmethod
    def check(cls, data: SpatialData) -> None:
        for axis in data.axes:
            coord = data.coordinates[axis]
            if len(coord) == 0:
                continue

            if coord.dtype.kind != "f":
                raise ValueError(
                    f"Invalid data type {coord.dtype} (expected a float dtype)"
                )
            dt = coord.dtype.type

            xmin, xmax = (dt(_) for _ in _AXES_LIMITS[axis])
            if (cmin := dt(np.min(coord))) < xmin or not math.isfinite(cmin):
                if math.isfinite(xmin):
                    hint = f"minimal value allowed is {xmin}"
                else:
                    assert xmin == -float("inf")
                    hint = "value must be finite"
                raise ValueError(
                    f"Invalid coordinate data for axis {axis!r} {cmin} ({hint})"
                )
            if (cmax := dt(np.max(coord))) > xmax or not math.isfinite(cmax):
                if math.isfinite(xmax):
                    hint = f"maximal value allowed is {xmax}"
                else:
                    assert xmax == float("inf")
                    hint = "value must be finite"
                raise ValueError(
                    f"Invalid coordinate data for axis {axis!r} {cmax} ({hint})"
                )


@final
class DTypeConsistencyValidator:
    @classmethod
    def check(cls, data: SpatialData) -> None:
        dts = {
            name: arr.dtype
            for name, arr in chain(
                data.coordinates.items(),
                data.fields.items(),
            )
        }
        unique_dts = sorted(set(dts.values()))
        if len(unique_dts) > 1:
            raise TypeError(f"Received mixed data types ({unique_dts}):\n{dts}")


@final
@dataclass(frozen=True, slots=True)
class NamedArray:
    name: Name
    data: NDArray


@final
class FieldMapsValidatorHelper:
    @classmethod
    def check(
        cls,
        *fmaps: FieldMap | None,
        require_shape_equality: bool = False,
        require_sorted: bool = False,
        required_attrs: dict[str, Any] | None = None,
    ) -> None:
        ref_arr: NamedArray | None = None
        for name, data in chain.from_iterable(
            fm.items() for fm in fmaps if fm is not None
        ):
            if require_shape_equality:
                ref_arr = cls._validate_shape_equality(name, data, ref_arr)
            if require_sorted:
                cls._validate_sorted_state(name, data)
            if required_attrs:
                cls._validate_required_attributes(name, data, required_attrs)

    @staticmethod
    def _validate_shape_equality(
        name: str,
        data: NDArray[Real],
        ref_arr: NamedArray | None,
    ) -> NamedArray:
        if ref_arr is not None and data.shape != ref_arr.data.shape:
            raise ValueError(
                f"Fields {name!r} and {ref_arr.name!r} "
                f"have mismatching shapes {data.shape} and {ref_arr.data.shape}"
            )
        return ref_arr or NamedArray(name, data)

    @staticmethod
    def _validate_sorted_state(name: str, data: NDArray[Real]) -> None:
        a = data[0]
        for i, b in enumerate(data[1:], start=1):
            if a > b:
                raise ValueError(
                    f"Field {name!r} is not properly sorted by ascending order. "
                    f"Got {a} (index {i-1}) > {b} (index {i})"
                )
            a = b

    @staticmethod
    def _validate_required_attributes(
        name: str,
        data: NDArray[Real],
        required_attrs: dict[str, Any],
    ) -> None:
        for attr, expected in required_attrs.items():
            if (actual := getattr(data, attr)) != expected:
                raise ValueError(
                    f"Field {name!r} has incorrect {attr} {actual} "
                    f"(expected {expected})"
                )
