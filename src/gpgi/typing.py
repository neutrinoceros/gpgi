r"""Define public type annotations and protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Protocol

from gpgi._typing import (
    D_contra,
    F_contra,
    FieldMap,  # re-export
)

if TYPE_CHECKING:
    from gpgi._typing import D0, D1, FArray, HCIArray


__all__ = [
    "DepositionMethodT",
    "DepositionMethodWithMetadataT",
    "FieldMap",
]


class DepositionMethodT(Generic[D_contra, F_contra], Protocol):
    def __call__(  # noqa D102
        self,
        cell_edges_x1: FArray[D1, F_contra],
        cell_edges_x2: FArray[D1, F_contra],
        cell_edges_x3: FArray[D1, F_contra],
        particles_x1: FArray[D1, F_contra],
        particles_x2: FArray[D1, F_contra],
        particles_x3: FArray[D1, F_contra],
        field: FArray[D1, F_contra],
        weight_field: FArray[D1, F_contra] | FArray[D0, F_contra],
        hci: HCIArray,
        out: FArray[D_contra, F_contra],
    ) -> None: ...


class DepositionMethodWithMetadataT(Generic[D_contra, F_contra], Protocol):
    def __call__(  # noqa D102
        self,
        cell_edges_x1: FArray[D1, F_contra],
        cell_edges_x2: FArray[D1, F_contra],
        cell_edges_x3: FArray[D1, F_contra],
        particles_x1: FArray[D1, F_contra],
        particles_x2: FArray[D1, F_contra],
        particles_x3: FArray[D1, F_contra],
        field: FArray[D1, F_contra],
        weight_field: FArray[D1, F_contra] | FArray[D0, F_contra],
        hci: HCIArray,
        out: FArray[D_contra, F_contra],
        *,
        metadata: dict[str, Any],
    ) -> None: ...
