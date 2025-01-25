r"""Define public type annotations and protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

# re-exports
from gpgi._typing import FieldMap

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from gpgi._typing import FloatT, HCIArray


__all__ = [
    "DepositionMethodT",
    "DepositionMethodWithMetadataT",
    "FieldMap",
]


class DepositionMethodT(Protocol):
    def __call__(  # noqa D102
        self,
        cell_edges_x1: NDArray[FloatT],
        cell_edges_x2: NDArray[FloatT],
        cell_edges_x3: NDArray[FloatT],
        particles_x1: NDArray[FloatT],
        particles_x2: NDArray[FloatT],
        particles_x3: NDArray[FloatT],
        field: NDArray[FloatT],
        weight_field: NDArray[FloatT],
        hci: HCIArray,
        out: NDArray[FloatT],
    ) -> None: ...


class DepositionMethodWithMetadataT(Protocol):
    def __call__(  # noqa D102
        self,
        cell_edges_x1: NDArray[FloatT],
        cell_edges_x2: NDArray[FloatT],
        cell_edges_x3: NDArray[FloatT],
        particles_x1: NDArray[FloatT],
        particles_x2: NDArray[FloatT],
        particles_x3: NDArray[FloatT],
        field: NDArray[FloatT],
        weight_field: NDArray[FloatT],
        hci: HCIArray,
        out: NDArray[FloatT],
        *,
        metadata: dict[str, Any],
    ) -> None: ...
