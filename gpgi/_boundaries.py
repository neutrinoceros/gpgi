from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, cast

if TYPE_CHECKING:
    from ._typing import RealArray

BoundaryRecipeT = Callable[
    [
        "RealArray",
        "RealArray",
        "RealArray",
        "RealArray",
        "RealArray",
        "RealArray",
        "RealArray",
        "RealArray",
        Literal["left", "right"],
        Dict[str, Any],
    ],
    "RealArray",
]


class BoundaryRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, BoundaryRecipeT] = {}
        for key, recipe in _base_registry.items():
            self.register(key, recipe, skip_validation=True)

    @staticmethod
    def _validate_recipe(recipe: BoundaryRecipeT) -> None:
        import inspect

        sig = inspect.signature(recipe)
        params = [
            name for name, p in sig.parameters.items() if p.default == inspect._empty
        ]
        if params != [
            "same_side_active_layer",
            "same_side_ghost_layer",
            "opposite_side_active_layer",
            "opposite_side_ghost_layer",
            "weight_same_side_active_layer",
            "weight_same_side_ghost_layer",
            "weight_opposite_side_active_layer",
            "weight_opposite_side_ghost_layer",
            "side",
            "metadata",
        ]:
            raise ValueError(
                "Invalid boundary recipe. Expected a function with exactly 6 parameters, "
                "named 'same_side_active_layer', 'same_side_ghost_layer', "
                "'opposite_side_active_layer', 'opposite_side_ghost_layer', "
                "'weight_same_side_active_layer', 'weight_same_side_ghost_layer', "
                "'weight_opposite_side_active_layer', 'weight_opposite_side_ghost_layer', "
                "'side', and 'metadata'"
            )

    def register(
        self, key: str, recipe: BoundaryRecipeT, *, skip_validation: bool = False
    ) -> None:
        if not skip_validation:
            self._validate_recipe(recipe)

        if key in self._registry:
            warnings.warn(f"Overriding existing method {key!r}", stacklevel=2)
        self._registry[key] = recipe

    def __getitem__(self, key: str) -> BoundaryRecipeT:
        return self._registry[key]

    def __contains__(self, key: str) -> bool:
        return key in self._registry


# basic recipes
def open_boundary(
    same_side_active_layer: RealArray,
    same_side_ghost_layer: RealArray,
    opposite_side_active_layer: RealArray,
    opposite_side_ghost_layer: RealArray,
    weight_same_side_active_layer: RealArray,
    weight_same_side_ghost_layer: RealArray,
    weight_opposite_side_active_layer: RealArray,
    weight_opposite_side_ghost_layer: RealArray,
    side: Literal["left", "right"],
    metadata: dict[str, Any],
) -> RealArray:
    # return the active layer unchanged
    return same_side_active_layer


def wall_boundary(
    same_side_active_layer: RealArray,
    same_side_ghost_layer: RealArray,
    opposite_side_active_layer: RealArray,
    opposite_side_ghost_layer: RealArray,
    weight_same_side_active_layer: RealArray,
    weight_same_side_ghost_layer: RealArray,
    weight_opposite_side_active_layer: RealArray,
    weight_opposite_side_ghost_layer: RealArray,
    side: Literal["left", "right"],
    metadata: dict[str, Any],
) -> RealArray:
    return cast("RealArray", same_side_active_layer + same_side_ghost_layer)


def antisymmetric_boundary(
    same_side_active_layer: RealArray,
    same_side_ghost_layer: RealArray,
    opposite_side_active_layer: RealArray,
    opposite_side_ghost_layer: RealArray,
    weight_same_side_active_layer: RealArray,
    weight_same_side_ghost_layer: RealArray,
    weight_opposite_side_active_layer: RealArray,
    weight_opposite_side_ghost_layer: RealArray,
    side: Literal["left", "right"],
    metadata: dict[str, Any],
) -> RealArray:
    return cast("RealArray", same_side_active_layer - same_side_ghost_layer)


def periodic_boundary(
    same_side_active_layer: RealArray,
    same_side_ghost_layer: RealArray,
    opposite_side_active_layer: RealArray,
    opposite_side_ghost_layer: RealArray,
    weight_same_side_active_layer: RealArray,
    weight_same_side_ghost_layer: RealArray,
    weight_opposite_side_active_layer: RealArray,
    weight_opposite_side_ghost_layer: RealArray,
    side: Literal["left", "right"],
    metadata: dict[str, Any],
) -> RealArray:
    return cast("RealArray", same_side_active_layer + opposite_side_ghost_layer)


_base_registry: dict[str, BoundaryRecipeT] = {
    "open": open_boundary,
    "wall": wall_boundary,
    "antisymmetric ": antisymmetric_boundary,
    "periodic": periodic_boundary,
}
