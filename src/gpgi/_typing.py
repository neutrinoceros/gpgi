from typing import Generic, NotRequired, TypedDict, TypeVar

import numpy as np
from numpy.typing import NDArray

RealT = TypeVar("RealT", np.float32, np.float64)
RealArray = NDArray[RealT]
HCIArray = NDArray[np.uint16]


Name = str
FieldMap = dict[str, NDArray[RealT]]


class CartesianCoordinates(TypedDict, Generic[RealT]):
    x: NDArray[RealT]
    y: NotRequired[NDArray[RealT]]
    z: NotRequired[NDArray[RealT]]


class CylindricalCoordinates(TypedDict, Generic[RealT]):
    radius: NDArray[RealT]
    azimuth: NotRequired[NDArray[RealT]]
    z: NotRequired[NDArray[RealT]]


class PolarCoordinates(TypedDict, Generic[RealT]):
    radius: NDArray[RealT]
    z: NotRequired[NDArray[RealT]]
    azimuth: NotRequired[NDArray[RealT]]


class SphericalCoordinates(TypedDict, Generic[RealT]):
    colatitude: NDArray[RealT]
    radius: NotRequired[NDArray[RealT]]
    azimuth: NotRequired[NDArray[RealT]]


class EquatorialCoordinates(TypedDict, Generic[RealT]):
    radius: NDArray[RealT]
    latitude: NotRequired[NDArray[RealT]]
    azimuth: NotRequired[NDArray[RealT]]


CoordMap = (
    CartesianCoordinates[RealT]
    | CylindricalCoordinates[RealT]
    | PolarCoordinates[RealT]
    | SphericalCoordinates[RealT]
    | EquatorialCoordinates[RealT]
)


class GridDict(TypedDict, Generic[RealT]):
    cell_edges: CoordMap[RealT]
    fields: NotRequired[FieldMap[RealT]]


class ParticleSetDict(TypedDict, Generic[RealT]):
    coordinates: CoordMap[RealT]
    fields: NotRequired[FieldMap[RealT]]
