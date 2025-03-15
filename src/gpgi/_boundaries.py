from collections.abc import Callable
from threading import Lock
from typing import Any, Literal, cast

from numpy.typing import NDArray

from gpgi._typing import FloatT

BoundaryRecipeT = Callable[
    [
        NDArray[FloatT],
        NDArray[FloatT],
        NDArray[FloatT],
        NDArray[FloatT],
        NDArray[FloatT],
        NDArray[FloatT],
        NDArray[FloatT],
        NDArray[FloatT],
        Literal["left", "right"],
        dict[str, Any],
    ],
    NDArray[FloatT],
]


class BoundaryRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, BoundaryRecipeT] = {}
        self._lock = Lock()
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
        self,
        key: str,
        recipe: BoundaryRecipeT,
        *,
        skip_validation: bool = False,
        allow_unsafe_override: bool = False,
    ) -> None:
        """
        Register a new boundary function.

        Parameters
        ----------
        key: str
            A unique identifier (ideally a meaningful name) to associate with
            the function.

        recipe: Callable
            A function matching the signature (order and names of arguments) of
            gpgi's builtin boundary recipes.

        skip_validation: bool, optional, keyword-only (default: False)
            If set to True, signature validation is skipped.
            This is meant to allow bypassing hypothetical bugs in the validation
            routine.

        allow_unsafe_override: bool, optional, keyword-only (default: False)
            If set to True, registering a new function under an existing key
            will not raise an exception. Note however that doing so is not
            thread-safe.

        Raises
        ------
        ValueError:
        - if skip_validation==False and the signature of the recipe doesn't meet
          the requirements.
        - if allow_unsafe_override==False and a new function is being registered
          under an already used key. Registering the same exact function under
          multiple times either under the same key or another, unused key, is
          always safe so it does not raise.
        """
        with self._lock:
            if key in self._registry:
                if recipe is self._registry[key]:
                    return
                elif not allow_unsafe_override:
                    raise ValueError(
                        f"Another function is already registered with {key=!r}. "
                        "If you meant to override the existing function, "
                        "consider setting allow_unsafe_override=True"
                    )

            if not skip_validation:
                self._validate_recipe(recipe)

            self._registry[key] = recipe

    def __getitem__(self, key: str) -> BoundaryRecipeT:
        return self._registry[key]

    def __contains__(self, key: str) -> bool:
        return key in self._registry


# basic recipes
def open_boundary(
    same_side_active_layer: NDArray[FloatT],
    same_side_ghost_layer: NDArray[FloatT],
    opposite_side_active_layer: NDArray[FloatT],
    opposite_side_ghost_layer: NDArray[FloatT],
    weight_same_side_active_layer: NDArray[FloatT],
    weight_same_side_ghost_layer: NDArray[FloatT],
    weight_opposite_side_active_layer: NDArray[FloatT],
    weight_opposite_side_ghost_layer: NDArray[FloatT],
    side: Literal["left", "right"],
    metadata: dict[str, Any],
) -> NDArray[FloatT]:
    # return the active layer unchanged
    return same_side_active_layer


def wall_boundary(
    same_side_active_layer: NDArray[FloatT],
    same_side_ghost_layer: NDArray[FloatT],
    opposite_side_active_layer: NDArray[FloatT],
    opposite_side_ghost_layer: NDArray[FloatT],
    weight_same_side_active_layer: NDArray[FloatT],
    weight_same_side_ghost_layer: NDArray[FloatT],
    weight_opposite_side_active_layer: NDArray[FloatT],
    weight_opposite_side_ghost_layer: NDArray[FloatT],
    side: Literal["left", "right"],
    metadata: dict[str, Any],
) -> NDArray[FloatT]:
    return cast("NDArray[FloatT]", same_side_active_layer + same_side_ghost_layer)


def antisymmetric_boundary(
    same_side_active_layer: NDArray[FloatT],
    same_side_ghost_layer: NDArray[FloatT],
    opposite_side_active_layer: NDArray[FloatT],
    opposite_side_ghost_layer: NDArray[FloatT],
    weight_same_side_active_layer: NDArray[FloatT],
    weight_same_side_ghost_layer: NDArray[FloatT],
    weight_opposite_side_active_layer: NDArray[FloatT],
    weight_opposite_side_ghost_layer: NDArray[FloatT],
    side: Literal["left", "right"],
    metadata: dict[str, Any],
) -> NDArray[FloatT]:
    return cast("NDArray[FloatT]", same_side_active_layer - same_side_ghost_layer)


def periodic_boundary(
    same_side_active_layer: NDArray[FloatT],
    same_side_ghost_layer: NDArray[FloatT],
    opposite_side_active_layer: NDArray[FloatT],
    opposite_side_ghost_layer: NDArray[FloatT],
    weight_same_side_active_layer: NDArray[FloatT],
    weight_same_side_ghost_layer: NDArray[FloatT],
    weight_opposite_side_active_layer: NDArray[FloatT],
    weight_opposite_side_ghost_layer: NDArray[FloatT],
    side: Literal["left", "right"],
    metadata: dict[str, Any],
) -> NDArray[FloatT]:
    return cast("NDArray[FloatT]", same_side_active_layer + opposite_side_ghost_layer)


_base_registry: dict[str, BoundaryRecipeT] = {
    "open": open_boundary,
    "wall": wall_boundary,
    "antisymmetric ": antisymmetric_boundary,
    "periodic": periodic_boundary,
}
