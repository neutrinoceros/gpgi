import pytest

from gpgi._boundaries import BoundaryRegistry


def test_boundary_register_overrides():
    registry = BoundaryRegistry()

    def test_recipe1(
        same_side_active_layer,
        same_side_ghost_layer,
        opposite_side_active_layer,
        opposite_side_ghost_layer,
        weight_same_side_active_layer,
        weight_same_side_ghost_layer,
        weight_opposite_side_active_layer,
        weight_opposite_side_ghost_layer,
        side,
        metadata,
    ): ...
    def test_recipe2(
        same_side_active_layer,
        same_side_ghost_layer,
        opposite_side_active_layer,
        opposite_side_ghost_layer,
        weight_same_side_active_layer,
        weight_same_side_ghost_layer,
        weight_opposite_side_active_layer,
        weight_opposite_side_ghost_layer,
        side,
        metadata,
    ): ...

    registry.register("test1", test_recipe1)
    assert registry["test1"] is test_recipe1

    # registering the same function a second time shouldn't raise
    registry.register("test1", test_recipe1)

    with pytest.raises(
        ValueError,
        match="Another function is already registered with key='test1'",
    ):
        registry.register("test1", test_recipe2)

    # check that we raised in time to preserve state
    assert registry["test1"] is test_recipe1

    # if we explicitly allow unsafe mutations, this should not raise
    registry.register("test1", test_recipe2, allow_unsafe_override=True)
    assert registry["test1"] is test_recipe2
