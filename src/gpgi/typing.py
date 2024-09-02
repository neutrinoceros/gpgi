r"""Define public type annotations and protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

# re-exports
from gpgi._typing import FieldMap

if TYPE_CHECKING:
    from gpgi._typing import HCIArray, RealArray


__all__ = [
    "DepositionMethodT",
    "DepositionMethodWithMetadataT",
    "FieldMap",
]


class DepositionMethodT(Protocol):
    def __call__(  # noqa D102
        self,
        cell_edges_x1: RealArray,
        cell_edges_x2: RealArray,
        cell_edges_x3: RealArray,
        particles_x1: RealArray,
        particles_x2: RealArray,
        particles_x3: RealArray,
        field: RealArray,
        weight_field: RealArray,
        hci: HCIArray,
        out: RealArray,
    ) -> None: ...


class DepositionMethodWithMetadataT(Protocol):
    def __call__(  # noqa D102
        self,
        cell_edges_x1: RealArray,
        cell_edges_x2: RealArray,
        cell_edges_x3: RealArray,
        particles_x1: RealArray,
        particles_x2: RealArray,
        particles_x3: RealArray,
        field: RealArray,
        weight_field: RealArray,
        hci: HCIArray,
        out: RealArray,
        *,
        metadata: dict[str, Any],
    ) -> None: ...
