import re

import numpy as np
import pytest
from pytest import RaisesExc, RaisesGroup

import gpgi


def test_load_null_dataset():
    with pytest.raises(TypeError):
        gpgi.load(geometry="cartesian")


def test_load_standalone_grid():
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.array([0.0, 1.0]),
                "y": np.array([0.0, 1]),
                "z": np.array([0.0, 1.0]),
            }
        },
    )
    assert ds.grid.ndim == 3
    assert ds.grid.axes == ("x", "y", "z")
    assert ds.grid.shape == (1, 1, 1)
    assert ds.particles.count == 0
    assert ds.metadata == {}


def test_metadata():
    md = {"time": 0.1}
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.array([-10.0, 10.0]),
                "y": np.array([-10.0, 10.0]),
                "z": np.array([-10.0, 10.0]),
            }
        },
        particles={
            "coordinates": {
                "x": np.array([0.0, 1.0]),
                "y": np.array([0.0, 1.0]),
                "z": np.array([0.0, 1.0]),
            }
        },
        metadata=md,
    )
    assert ds.grid.ndim == ds.particles.ndim == 3
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
        gpgi.load(
            geometry="cartesian",
            grid={"cell_edges": {"x": np.array([0.0, 1.0])}},
            particles={},
        )


def test_load_empty_grid_and_empty_particles():
    with RaisesGroup(
        RaisesExc(
            ValueError,
            match="grid dictionary missing required key 'cell_edges'",
        ),
        RaisesExc(
            ValueError,
            match="particles dictionary missing required key 'coordinates'",
        ),
        match=r"^Invalid inputs were received$",
    ):
        gpgi.load(geometry="cartesian", grid={}, particles={})


def test_unsorted_cell_edges():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Field 'x' is not properly sorted by ascending order. "
            r"Got 1.0 (index 0) > 0.0 (index 1)"
        ),
    ):
        gpgi.load(
            geometry="cartesian", grid={"cell_edges": {"x": np.array([1.0, 0.0])}}
        )


@pytest.mark.parametrize(
    "side, out_of_bounds_value",
    [
        ("min", -float("inf")),
        ("max", float("inf")),
    ],
)
def test_infinite_box_edges(side, out_of_bounds_value):
    if side == "min":
        xlim = np.array([-np.inf, 1.0])
    elif side == "max":
        xlim = np.array([-0.1, +np.inf])
    with pytest.raises(
        ValueError,
        match=(
            f"Invalid coordinate data for axis 'x' {out_of_bounds_value} "
            rf"\(value must be finite\)"
        ),
    ):
        gpgi.load(geometry="cartesian", grid={"cell_edges": {"x": xlim}})


@pytest.mark.parametrize("dtype", ["int16", "int32", "int64"])
def test_integer_dtype(dtype):
    with pytest.raises(
        ValueError,
        match=f"Invalid data type {dtype}",
    ):
        gpgi.load(
            geometry="cartesian",
            grid={
                "cell_edges": {"x": np.arange(10, dtype=dtype)},
            },
        )


def test_missing_grid():
    with pytest.raises(
        TypeError, match=r"load\(\) missing 1 required keyword-only argument: 'grid'"
    ):
        gpgi.load(
            geometry="cartesian",
            particles={
                "coordinates": {"x": np.arange(10, dtype="float64")},
                "fields": {"mass": np.ones(10, dtype="float64")},
            },
        )


@pytest.mark.parametrize(
    "geometry, cell_edges, coords, axis, side, limit",
    [
        pytest.param(
            "polar",
            {"radius": np.array([0.0, 1.0])},
            {"radius": np.arange(-1.0, 1.0)},
            "radius",
            "min",
            0.0,
            id="polar_min_radius",
        ),
        pytest.param(
            "cylindrical",
            {"radius": np.array([0.0, 1.0])},
            {"radius": np.arange(-1.0, 1.0)},
            "radius",
            "min",
            0.0,
            id="cylindrical_min_radius",
        ),
        pytest.param(
            "spherical",
            {"radius": np.array([0.0, 1.0])},
            {"radius": np.arange(-1.0, 1.0)},
            "radius",
            "min",
            0.0,
            id="spherical_min_radius",
        ),
        pytest.param(
            "cylindrical",
            {
                "radius": np.array([0.0, 100.0]),
                "z": np.array([0.0, 100.0]),
                "azimuth": np.array([0.0, 2 * np.pi]),
            },
            {
                "radius": np.arange(10.0),
                "z": np.arange(10.0),
                "azimuth": np.arange(-10.0, 0.0),
            },
            "azimuth",
            "min",
            0.0,
            id="cylindrical_min_azimuth",
        ),
        pytest.param(
            "cylindrical",
            {
                "radius": np.array([0.0, 100.0]),
                "z": np.array([0.0, 100.0]),
                "azimuth": np.array([0.0, 2 * np.pi]),
            },
            {
                "radius": np.arange(10.0),
                "z": np.arange(10.0),
                "azimuth": np.arange(10.0),
            },
            "azimuth",
            "max",
            2 * np.pi,
            id="cylindrical_max_azimuth",
        ),
        pytest.param(
            "spherical",
            {"radius": np.array([0.0, 100.0]), "colatitude": np.array([0.0, np.pi])},
            {"radius": np.arange(10.0), "colatitude": np.arange(-10.0, 0.0)},
            "colatitude",
            "min",
            0.0,
            id="spherical_min_colatitude",
        ),
        pytest.param(
            "spherical",
            {"radius": np.array([0.0, 100.0]), "colatitude": np.array([0.0, np.pi])},
            {"radius": np.arange(10.0), "colatitude": np.arange(10.0)},
            "colatitude",
            "max",
            np.pi,
            id="spherical_max_colatitude",
        ),
        pytest.param(
            "spherical",
            {
                "radius": np.array([0.0, 100.0]),
                "colatitude": np.array([0.0, np.pi]),
                "azimuth": np.array([0, 2 * np.pi]),
            },
            {
                "radius": np.arange(10.0),
                "colatitude": np.linspace(0.0, np.pi, 10),
                "azimuth": np.arange(-10.0, 0.0),
            },
            "azimuth",
            "min",
            0.0,
            id="spherical_min_azimuth",
        ),
        pytest.param(
            "spherical",
            {
                "radius": np.array([0.0, 100.0]),
                "colatitude": np.array([0.0, np.pi]),
                "azimuth": np.array([0, 2 * np.pi]),
            },
            {
                "radius": np.arange(10.0),
                "colatitude": np.linspace(0.0, np.pi, 10),
                "azimuth": np.arange(10.0),
            },
            "azimuth",
            "max",
            2 * np.pi,
            id="spherical_max_azimuth",
        ),
        pytest.param(
            "equatorial",
            {
                "radius": np.array([0.0, 100.0]),
                "azimuth": np.array([0, 2 * np.pi]),
                "latitude": np.array([-np.pi / 2, np.pi / 2]),
            },
            {
                "radius": np.arange(10.0),
                "azimuth": np.linspace(0.0, 2 * np.pi, 10),
                "latitude": np.arange(-10.0, 0.0),
            },
            "latitude",
            "min",
            -np.pi / 2,
            id="equatorial_min_latitude",
        ),
        pytest.param(
            "equatorial",
            {
                "radius": np.array([0.0, 100.0]),
                "azimuth": np.array([0, 2 * np.pi]),
                "latitude": np.array([-np.pi / 2, np.pi / 2]),
            },
            {
                "radius": np.arange(10.0),
                "azimuth": np.arange(10.0),
                "latitude": np.linspace(-np.pi / 2, np.pi / 2, 10),
            },
            "azimuth",
            "max",
            2 * np.pi,
            id="equatorial_max_azimuth",
        ),
    ],
)
def test_load_invalid_particles_coordinates(
    geometry, cell_edges, coords, axis, side, limit
):
    dt = coords[list(coords.keys())[0]].dtype.type
    if side == "min":
        c = dt(coords[axis].min())
    elif side == "max":
        c = dt(coords[axis].max())
    with pytest.raises(
        ValueError,
        match=(
            f"Invalid coordinate data for axis {axis!r} {c} "
            rf"\({side}imal value allowed is {limit}\)"
        ),
    ):
        gpgi.load(
            geometry=geometry,
            grid={"cell_edges": cell_edges},
            particles={"coordinates": coords},
        )


def test_multiple_invalid_particle_coordinates():
    cell_edges = {
        "radius": np.array([0.0, 100.0]),
        "colatitude": np.array([0.0, np.pi]),
        "azimuth": np.array([0, 2 * np.pi]),
    }
    coords = {
        "radius": np.arange(10.0),
        "colatitude": np.linspace(-0.1, np.pi + 0.1, 10),
        "azimuth": np.linspace(-0.1, 2 * np.pi + 0.1, 10),
    }
    dt = coords[list(coords.keys())[0]].dtype.type

    expected_exceptions = [
        RaisesExc(
            ValueError,
            match=(
                f"Invalid coordinate data for axis {axis!r} {dt(getattr(coords[axis], side)())} "
                rf"\({side}imal value allowed is {limit}\)"
            ),
        )
        for axis, side, limit in [
            ("colatitude", "min", 0.0),
            ("colatitude", "max", np.pi),
            ("azimuth", "min", 0.0),
            ("azimuth", "max", 2 * np.pi),
        ]
    ]

    with RaisesGroup(*expected_exceptions, match="input particle data is invalid"):
        gpgi.load(
            geometry="spherical",
            grid={"cell_edges": cell_edges},
            particles={"coordinates": coords},
        )


def test_inconsistent_shape_particle_data():
    with pytest.raises(
        ValueError,
        match=re.escape(r"Fields 'y' and 'x' have mismatching shapes (3,) and (2,)"),
    ):
        gpgi.load(
            geometry="cartesian",
            grid={
                "cell_edges": {
                    "x": np.array([-1.0, 1.0]),
                    "y": np.array([-1.0, 1.0]),
                },
            },
            particles={
                "coordinates": {
                    "x": np.array([0.0, 0.0]),
                    "y": np.array([0.0, 0.0, 0.0]),
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
                    "x": np.array([0.0, 0.0]),
                    "y": np.array([0.0, 0.0, 0.0]),
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
                "x": np.array([0.0, 0.0]),
                "y": np.array([0.0, 0.0, 0.0]),
            },
            "fields": {},
        },
    )
    assert ds.grid.fields == {}


def position(i):
    return ["first", "second", "third"][i]


@pytest.mark.parametrize(
    "geometry, axes, pos",
    [
        ("cartesian", ("x", "azimuth"), 1),
        ("cartesian", ("x", "y", "colatitude"), 2),
        ("cartesian", ("z",), 0),
        ("cartesian", ("y",), 0),
        ("spherical", ("x",), 0),
    ],
)
def test_single_invalid_axis(geometry, axes, pos):
    with pytest.raises(
        ValueError,
        match=rf"Invalid {position(pos)} axis name '\w+', with geometry {geometry!r}",
    ):
        gpgi.load(
            geometry=geometry,
            grid={"cell_edges": {_: np.array([0.0, 1.0]) for _ in axes}},
        )


@pytest.mark.parametrize(
    "geometry, axes, positions",
    [
        ("spherical", ("x", "y"), [0, 1]),
        ("spherical", ("x", "y", "z"), [0, 1, 2]),
        ("spherical", ("radius", "azimuth", "z"), [1, 2]),
    ],
)
def test_multiple_invalid_axes(geometry, axes, positions):
    with RaisesGroup(
        *[
            RaisesExc(
                ValueError,
                match=rf"Invalid {position(pos)} axis name '\w+', with geometry {geometry!r}",
            )
            for pos in positions
        ],
        match="input grid data is invalid",
    ):
        gpgi.load(
            geometry=geometry,
            grid={"cell_edges": {_: np.array([0.0, 1.0]) for _ in axes}},
        )


def test_invalid_geometry():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "unknown geometry 'unknown', expected any of "
            r"('cartesian', 'polar', 'cylindrical', 'spherical', 'equatorial')"
        ),
    ):
        gpgi.load(geometry="unknown", grid={"cell_edges": {"ax": np.array([0.0, 1.0])}})


def test_out_of_bound_particles_left():
    with pytest.raises(
        ValueError, match="Got particle at radius=1.0 < domain_left=10.0"
    ):
        gpgi.load(
            geometry="spherical",
            grid={"cell_edges": {"radius": np.arange(10, 101, dtype="float64")}},
            particles={"coordinates": {"radius": np.array([1.0])}},
        )


def test_out_of_bound_particles_right():
    with pytest.raises(
        ValueError, match="Got particle at radius=1000.0 > domain_right=100.0"
    ):
        gpgi.load(
            geometry="spherical",
            grid={"cell_edges": {"radius": np.arange(10, 101, dtype="float64")}},
            particles={"coordinates": {"radius": np.array([1000.0])}},
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
    with pytest.raises(TypeError, match=r"Received mixed data types"):
        gpgi.load(
            geometry="polar",
            grid={
                "cell_edges": {
                    "radius": np.array([0, 1], dtype="float64"),
                    "azimuth": np.array([0, 2 * np.pi], dtype="float32"),
                }
            },
        )
