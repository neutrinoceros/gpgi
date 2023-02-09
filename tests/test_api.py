import re

import numpy as np
import pytest

import gpgi


def test_load_null_dataset():
    with pytest.raises(
        TypeError,
        match=(
            "Cannot instantiate empty dataset. "
            "Grid and/or particle data must be provided"
        ),
    ):
        gpgi.load(geometry="cartesian")


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
    assert ds.particles.count == 0
    assert ds.metadata == {}


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
    assert ds.grid.shape == (1, 1, 1)
    assert ds.metadata == {}


def test_metadata():
    md = {
        "time": 0.1,
    }
    ds = gpgi.load(
        geometry="cartesian",
        particles={
            "coordinates": {
                "x": np.array([0, 1]),
                "y": np.array([0, 1]),
                "z": np.array([0, 1]),
            }
        },
        metadata=md,
    )
    assert set(ds.metadata.items()) >= set(md.items())
    assert ds.metadata == md
    assert ds.metadata is not md


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


def test_unsorted_cell_edges():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Field 'x' is not properly sorted by ascending order. "
            r"Got 1.0 (index 0) > 0.0 (index 1)"
        ),
    ):
        gpgi.load(geometry="cartesian", grid={"cell_edges": {"x": np.array([1, 0])}})


@pytest.mark.parametrize(
    "geometry, coords, axis, side, limit",
    [
        ("polar", {"radius": np.arange(-1.0, 1.0)}, "radius", "min", 0.0),
        ("cylindrical", {"radius": np.arange(-1.0, 1.0)}, "radius", "min", 0.0),
        ("spherical", {"radius": np.arange(-1.0, 1.0)}, "radius", "min", 0.0),
        (
            "cylindrical",
            {
                "radius": np.arange(10.0),
                "z": np.arange(10.0),
                "azimuth": np.arange(-10.0, 10.0),
            },
            "azimuth",
            "min",
            0.0,
        ),
        (
            "cylindrical",
            {
                "radius": np.arange(10.0),
                "z": np.arange(10.0),
                "azimuth": np.arange(10.0),
            },
            "azimuth",
            "max",
            2 * np.pi,
        ),
        (
            "spherical",
            {"radius": np.arange(10.0), "colatitude": np.arange(-10.0, 10.0)},
            "colatitude",
            "min",
            0.0,
        ),
        (
            "spherical",
            {"radius": np.arange(10.0), "colatitude": np.arange(10.0)},
            "colatitude",
            "max",
            np.pi,
        ),
        (
            "spherical",
            {
                "radius": np.arange(10.0),
                "colatitude": np.arange(2.0),
                "azimuth": np.arange(-10.0, 10.0),
            },
            "azimuth",
            "min",
            0.0,
        ),
        (
            "spherical",
            {
                "radius": np.arange(10.0),
                "colatitude": np.arange(2.0),
                "azimuth": np.arange(10.0),
            },
            "azimuth",
            "max",
            2 * np.pi,
        ),
    ],
)
def test_load_invalid_grid_coordinates(geometry, coords, axis, side, limit):
    dt = coords[list(coords.keys())[0]].dtype.type
    if side == "min":
        c = dt(coords[axis].min())
    elif side == "max":
        c = dt(coords[axis].max())
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
    "geometry, axes, pos",
    [
        ("cartesian", ("x", "azimuth"), 1),
        ("cartesian", ("x", "y", "colatitude"), 2),
        ("cartesian", ("z",), 0),
        ("cartesian", ("y",), 0),
        ("spherical", ("x", "y", "z"), 0),
        ("spherical", ("x", "y"), 0),
        ("spherical", ("x",), 0),
        ("spherical", ("radius", "azimuth", "z"), 1),
    ],
)
def test_invalid_axes(geometry, axes, pos):
    with pytest.raises(
        ValueError,
        match=rf"Got invalid axis name '\w+' on position {pos}, with geometry {geometry!r}",
    ):
        gpgi.load(
            geometry=geometry,
            grid={"cell_edges": {_: [0, 1] for _ in axes}},
        )


def test_invalid_geometry():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "unknown geometry 'unknown', expected any of "
            r"('cartesian', 'polar', 'cylindrical', 'spherical', 'equatorial')"
        ),
    ):
        gpgi.load(geometry="unknown")


def test_out_of_bound_particles_left():
    with pytest.raises(
        ValueError, match="Got particle at radius=1.0 < domain_left=10.0"
    ):
        gpgi.load(
            geometry="spherical",
            grid={"cell_edges": {"radius": np.arange(10, 101)}},
            particles={"coordinates": {"radius": np.array([1])}},
        )


def test_out_of_bound_particles_right():
    with pytest.raises(
        ValueError, match="Got particle at radius=1000.0 > domain_right=100.0"
    ):
        gpgi.load(
            geometry="spherical",
            grid={"cell_edges": {"radius": np.arange(10, 101)}},
            particles={"coordinates": {"radius": np.array([1000])}},
        )


def test_identical_dtype_requirement():
    nx = 64
    nparticles = 600

    with pytest.raises(
        TypeError,
        match=r"Got mixed data types",
    ):
        gpgi.load(
            geometry="cartesian",
            grid={
                "cell_edges": {
                    "x": np.linspace(-1, 1, nx, dtype="float32"),
                },
            },
            particles={
                "coordinates": {
                    "x": np.zeros(nparticles, dtype="float64"),
                },
            },
        )


def test_depr_pic():
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.array([0, 1.0]),
                "y": np.array([0, 1.0]),
                "z": np.array([0, 1.0]),
            }
        },
        particles={
            "coordinates": {
                "x": np.array([0.0, 1.0]),
                "y": np.array([0.0, 1.0]),
                "z": np.array([0.0, 1.0]),
            },
            "fields": {
                "mass": np.ones(2),
            },
        },
    )
    with pytest.warns(
        DeprecationWarning,
        match=(
            r"method='pic' is a deprecated alias for method='ngp', "
            r"please use 'ngp' \(or 'nearest_grid_point'\) directly"
        ),
    ):
        ds.deposit("mass", method="pic")


def test_float32_limit_validation():
    gpgi.load(
        geometry="polar",
        grid={
            "cell_edges": {
                "radius": np.array([0, 1], dtype="float32"),
                "azimuth": np.array([0, 2 * np.pi], dtype="float32"),
            }
        },
    )


def test_float32_limit_invalidation():
    # domain boundaries are validated before dtype consistency,
    # we want to make sure that this fails *despite* being
    # knowingly tolerant with the dtype of each individual
    # coordinate array as we validate boundaries
    with pytest.raises(TypeError, match=r"Grid received mixed data types"):
        gpgi.load(
            geometry="polar",
            grid={
                "cell_edges": {
                    "radius": np.array([0, 1], dtype="float64"),
                    "azimuth": np.array([0, 2 * np.pi], dtype="float32"),
                }
            },
        )
