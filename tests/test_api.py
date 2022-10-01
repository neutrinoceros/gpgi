import re

import numpy as np
import pytest

import gpgi


def test_load_null_dataset():
    ds = gpgi.load(geometry="cartesian")
    assert ds.grid is None
    assert ds.particles is None


def test_load_standalone_grid():
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.array([0, 1]),
                "y": np.array([0, 1]),
                "z": np.array([0, 1]),
            }
        },
    )
    assert ds.grid.ndim == 3
    assert ds.grid.axes == ("x", "y", "z")
    assert ds.grid.shape == (1, 1, 1)
    assert ds.particles is None


def test_load_standalone_particles():
    ds = gpgi.load(
        geometry="cartesian",
        particles={
            "coordinates": {
                "x": np.array([0, 1]),
                "y": np.array([0, 1]),
                "z": np.array([0, 1]),
            }
        },
    )
    assert ds.particles.ndim == 3
    assert ds.particles.axes == ("x", "y", "z")
    assert ds.particles.count == 2
    assert ds.grid is None


def test_load_empty_grid():
    with pytest.raises(
        ValueError, match="grid dictionary missing required key 'cell_edges'"
    ):
        gpgi.load(geometry="cartesian", grid={})


def test_load_empty_particles():
    with pytest.raises(
        ValueError, match="particles dictionary missing required key 'coordinates'"
    ):
        gpgi.load(geometry="cartesian", particles={})


@pytest.mark.parametrize(
    "geometry, coords, axis, side, limit",
    [
        ("polar", {"radius": np.arange(-1, 1)}, "radius", "min", 0),
        ("cylindrical", {"radius": np.arange(-1, 1)}, "radius", "min", 0),
        ("spherical", {"radius": np.arange(-1, 1)}, "radius", "min", 0),
        (
            "cylindrical",
            {"radius": np.arange(10), "azimuth": np.arange(-10, 10)},
            "azimuth",
            "min",
            0,
        ),
        (
            "cylindrical",
            {"radius": np.arange(10), "azimuth": np.arange(10)},
            "azimuth",
            "max",
            2 * np.pi,
        ),
        (
            "spherical",
            {"radius": np.arange(10), "colatitude": np.arange(-10, 10)},
            "colatitude",
            "min",
            0,
        ),
        (
            "spherical",
            {"radius": np.arange(10), "colatitude": np.arange(10)},
            "colatitude",
            "max",
            np.pi,
        ),
        (
            "spherical",
            {
                "radius": np.arange(10),
                "colatitude": np.arange(2),
                "azimuth": np.arange(-10, 10),
            },
            "azimuth",
            "min",
            0,
        ),
        (
            "spherical",
            {
                "radius": np.arange(10),
                "colatitude": np.arange(2),
                "azimuth": np.arange(10),
            },
            "azimuth",
            "max",
            2 * np.pi,
        ),
    ],
)
def test_load_invalid_grid_coordinates(geometry, coords, axis, side, limit):
    if side == "min":
        c = coords[axis].min()
    elif side == "max":
        c = coords[axis].max()
    else:
        raise ValueError("broken test case")
    with pytest.raises(
        ValueError,
        match=(
            f"Invalid coordinate data for axis {axis!r} {c} "
            rf"\({side}imal allowed value is {limit}\)"
        ),
    ):
        gpgi.load(geometry=geometry, particles={"coordinates": coords})


def test_inconsistent_shape_particle_data():
    with pytest.raises(
        ValueError,
        match=re.escape(r"Fields 'y' and 'x' have mismatching shapes (3,) and (2,)"),
    ):
        gpgi.load(
            geometry="cartesian",
            particles={
                "coordinates": {
                    "x": np.array([0, 0]),
                    "y": np.array([0, 0, 0]),
                }
            },
        )


@pytest.mark.parametrize(
    "invalid_attr, data, expected",
    [
        ("size", np.ones(3), 2),
        ("ndim", np.ones(2), 2),
        ("shape", np.ones((2, 1)), (1, 2)),
    ],
)
def test_inconsistent_grid_data(data, invalid_attr, expected):
    actual = getattr(data, invalid_attr)
    with pytest.raises(
        ValueError,
        match=re.escape(
            rf"Field 'density' has incorrect {invalid_attr} {actual} (expected {expected})"
        ),
    ):
        gpgi.load(
            geometry="cartesian",
            grid={
                "cell_edges": {
                    "x": np.array([0, 0]),
                    "y": np.array([0, 0, 0]),
                },
                "fields": {
                    "density": data,
                },
            },
        )


def test_validate_empty_fields():
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.array([0, 0]),
                "y": np.array([0, 0, 0]),
            },
            "fields": {},
        },
    )
    assert ds.grid.fields == {}


@pytest.mark.parametrize(
    "geometry, axes",
    [
        ("cartesian", ("x", "azimuth")),
        ("cartesian", ("x", "y", "colatitude")),
        ("cartesian", ("z",)),
        ("cartesian", ("y",)),
        ("spherical", ("x", "y", "z")),
        ("spherical", ("x", "y")),
        ("spherical", ("x",)),
        ("spherical", ("radius", "azimuth", "z")),
    ],
)
def test_invalid_axes(geometry, axes):
    with pytest.raises(
        ValueError, match=rf"Got invalid axis '\w+' with geometry {geometry!r}"
    ):
        gpgi.load(
            geometry=geometry,
            grid={"cell_edges": {_: [0, 1] for _ in axes}},
        )


def test_invalid_geometry():
    with pytest.raises(
        ValueError,
        match=re.escape(
            r"unknown geometry 'unknown', expected any of ('cartesian', 'polar', 'cylindrical', 'spherical')"
        ),
    ):
        gpgi.load(geometry="unknown")
