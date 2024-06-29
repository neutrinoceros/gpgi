import sys
from typing import TypedDict, TypeVar

import numpy as np
import numpy.typing as npt

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired

Real = TypeVar("Real", np.float32, np.float64)
RealArray = npt.NDArray[Real]
HCIArray = npt.NDArray[np.uint16]


Name = str
FieldMap = dict[Name, np.ndarray]


class _GridDict(TypedDict):
    cell_edges: FieldMap
    fields: NotRequired[FieldMap]


class _ParticleSetDict(TypedDict):
    coordinates: FieldMap
    fields: NotRequired[FieldMap]
