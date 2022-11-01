# requires numpy >= 1.21
from typing import TypeVar

import numpy as np
import numpy.typing as npt

Real = TypeVar("Real", np.float32, np.float64)
RealArray = npt.NDArray[Real]
HCIArray = npt.NDArray[np.uint16]
