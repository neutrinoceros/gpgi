from typing import NotRequired, TypedDict, TypeVar

import numpy as np
import numpy.typing as npt

Real = TypeVar("Real", np.float32, np.float64)
RealArray = npt.NDArray[Real]
HCIArray = npt.NDArray[np.uint16]


Name = str
FieldMap = dict[str, np.ndarray]


class CartesianCoordinates(TypedDict):
    x: np.ndarray
    y: NotRequired[np.ndarray]
    z: NotRequired[np.ndarray]


class CylindricalCoordinates(TypedDict):
    radius: np.ndarray
    azimuth: NotRequired[np.ndarray]
    z: NotRequired[np.ndarray]


class PolarCoordinates(TypedDict):
    radius: np.ndarray
    z: NotRequired[np.ndarray]
    azimuth: NotRequired[np.ndarray]


class SphericalCoordinates(TypedDict):
    colatitude: np.ndarray
    radius: NotRequired[np.ndarray]
    azimuth: NotRequired[np.ndarray]


class EquatorialCoordinates(TypedDict):
    radius: np.ndarray
    latitude: NotRequired[np.ndarray]
    azimuth: NotRequired[np.ndarray]


CoordMap = (
    CartesianCoordinates
    | CylindricalCoordinates
    | PolarCoordinates
    | SphericalCoordinates
    | EquatorialCoordinates
)


class GridDict(TypedDict):
    cell_edges: CoordMap
    fields: NotRequired[FieldMap]


class ParticleSetDict(TypedDict):
    coordinates: CoordMap
    fields: NotRequired[FieldMap]
