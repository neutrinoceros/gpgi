"""gpgi: Fast particle deposition at post-processing time."""

from importlib.util import find_spec

from ._data_types import Dataset, Geometry, Grid, ParticleSet
from ._load import load

__all__ = [
    "load",
    "Dataset",
    "Geometry",
    "Grid",
    "ParticleSet",
]

_IS_PY_LIB = find_spec("gpgi._lib").origin.endswith(".py")  # type: ignore [union-attr] # pyright: ignore [reportOptionalMemberAccess]
