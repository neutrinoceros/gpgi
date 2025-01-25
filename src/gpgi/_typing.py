from typing import Generic, NotRequired, TypedDict, TypeVar

import numpy as np
from numpy.typing import NDArray

FloatT = TypeVar("FloatT", np.float32, np.float64)
HCIArray = NDArray[np.uint16]


Name = str
FieldMap = dict[str, NDArray[FloatT]]


class CartesianCoordinates(TypedDict, Generic[FloatT]):
    x: NDArray[FloatT]
    y: NotRequired[NDArray[FloatT]]
    z: NotRequired[NDArray[FloatT]]


class CylindricalCoordinates(TypedDict, Generic[FloatT]):
    radius: NDArray[FloatT]
    azimuth: NotRequired[NDArray[FloatT]]
    z: NotRequired[NDArray[FloatT]]


class PolarCoordinates(TypedDict, Generic[FloatT]):
    radius: NDArray[FloatT]
    z: NotRequired[NDArray[FloatT]]
    azimuth: NotRequired[NDArray[FloatT]]


class SphericalCoordinates(TypedDict, Generic[FloatT]):
    colatitude: NDArray[FloatT]
    radius: NotRequired[NDArray[FloatT]]
    azimuth: NotRequired[NDArray[FloatT]]


class EquatorialCoordinates(TypedDict, Generic[FloatT]):
    radius: NDArray[FloatT]
    latitude: NotRequired[NDArray[FloatT]]
    azimuth: NotRequired[NDArray[FloatT]]


CoordMap = (
    CartesianCoordinates[FloatT]
    | CylindricalCoordinates[FloatT]
    | PolarCoordinates[FloatT]
    | SphericalCoordinates[FloatT]
    | EquatorialCoordinates[FloatT]
)


class GridDict(TypedDict, Generic[FloatT]):
    cell_edges: CoordMap[FloatT]
    fields: NotRequired[FieldMap[FloatT]]


class ParticleSetDict(TypedDict, Generic[FloatT]):
    coordinates: CoordMap[FloatT]
    fields: NotRequired[FieldMap[FloatT]]
