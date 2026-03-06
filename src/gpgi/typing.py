r"""Define public type annotations and protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

# re-exports
from gpgi._typing import FieldMap

if TYPE_CHECKING:
    from gpgi._typing import D0, D1, D, F, FArray, HCIArray


__all__ = [
    "DepositionMethodT",
    "DepositionMethodWithMetadataT",
    "FieldMap",
]


class DepositionMethodT(Protocol):
    def __call__(  # noqa D102
        self,
        cell_edges_x1: FArray[D1, F],
        cell_edges_x2: FArray[D1, F],
        cell_edges_x3: FArray[D1, F],
        particles_x1: FArray[D1, F],
        particles_x2: FArray[D1, F],
        particles_x3: FArray[D1, F],
        field: FArray[D, F],
        weight_field: FArray[D, F] | FArray[D0, F],
        hci: HCIArray,
        out: FArray[D, F],
    ) -> None: ...


class DepositionMethodWithMetadataT(Protocol):
    def __call__(  # noqa D102
        self,
        cell_edges_x1: FArray[D1, F],
        cell_edges_x2: FArray[D1, F],
        cell_edges_x3: FArray[D1, F],
        particles_x1: FArray[D1, F],
        particles_x2: FArray[D1, F],
        particles_x3: FArray[D1, F],
        field: FArray[D, F],
        weight_field: FArray[D, F] | FArray[D0, F],
        hci: HCIArray,
        out: FArray[D, F],
        *,
        metadata: dict[str, Any],
    ) -> None: ...
