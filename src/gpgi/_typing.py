__all__ = ["D", "D_contra", "D0", "D1", "D2", "D3", "DH", "F", "F_contra", "FArray"]
from typing import Generic, NotRequired, TypedDict, TypeVar

import numpy as np
from numpy import float32 as f32, float64 as f64

D0 = tuple[()]
D1 = tuple[int]
D2 = tuple[int, int]
D3 = tuple[int, int, int]
D = TypeVar("D", D0, D1, D2, D3)
D_contra = TypeVar("D_contra", D0, D1, D2, D3, contravariant=True)
DH = TypeVar("DH", D0, D1, D2)  # dim hyperplanes
F = TypeVar("F", f32, f64)
F_contra = TypeVar("F_contra", f32, f64, contravariant=True)
DT = TypeVar("DT", bound=np.dtype)

Array = np.ndarray[D, DT]
FArray = np.ndarray[D, np.dtype[F]]
HCIArray = Array[D, np.dtype[np.uint16]]


Name = str
FieldMap = dict[str, FArray[D, F]]


class CartesianCoordinates(TypedDict, Generic[F]):
    x: FArray[D1, F]
    y: NotRequired[FArray[D1, F]]
    z: NotRequired[FArray[D1, F]]


class CylindricalCoordinates(TypedDict, Generic[F]):
    radius: FArray[D1, F]
    azimuth: NotRequired[FArray[D1, F]]
    z: NotRequired[FArray[D1, F]]


class PolarCoordinates(TypedDict, Generic[F]):
    radius: FArray[D1, F]
    z: NotRequired[FArray[D1, F]]
    azimuth: NotRequired[FArray[D1, F]]


class SphericalCoordinates(TypedDict, Generic[F]):
    colatitude: FArray[D1, F]
    radius: NotRequired[FArray[D1, F]]
    azimuth: NotRequired[FArray[D1, F]]


class EquatorialCoordinates(TypedDict, Generic[F]):
    radius: FArray[D1, F]
    latitude: NotRequired[FArray[D1, F]]
    azimuth: NotRequired[FArray[D1, F]]


CoordMap = (
    CartesianCoordinates[F]
    | CylindricalCoordinates[F]
    | PolarCoordinates[F]
    | SphericalCoordinates[F]
    | EquatorialCoordinates[F]
)


class GridDict(TypedDict, Generic[D, F]):
    cell_edges: CoordMap[F]
    fields: NotRequired[FieldMap[D, F]]


class ParticleSetDict(TypedDict, Generic[D, F]):
    coordinates: CoordMap[F]
    fields: NotRequired[FieldMap[D, F]]
