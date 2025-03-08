import re
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from threading import Lock

import numpy as np
import numpy.testing as npt
import pytest

import gpgi
from gpgi._lib import _deposit_ngp_2D


@pytest.fixture()
def sample_2D_dataset():
    nparticles = 100
    nx, ny = grid_shape = 16, 16
    prng = np.random.RandomState(0)

    return gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.linspace(-1, 1, nx + 1),
                "y": np.linspace(0, 2, ny + 1),
            },
            "fields": {
                "density": np.ones(grid_shape),
            },
        },
        particles={
            "coordinates": {
                "x": (2 * prng.random_sample(nparticles) - 1),
                "y": (2 * prng.random_sample(nparticles)),
            },
            "fields": {
                "mass": np.ones(nparticles),
                "vx": (2 * prng.random_sample(nparticles) - 1),
                "vy": (2 * prng.random_sample(nparticles) - 1),
            },
        },
    )


@pytest.mark.parametrize("method", ["ngp", "cic", "tsc"])
def test_single_cell_grid(method):
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.array([-1.0, 1.0]),
            },
        },
        particles={
            "coordinates": {"x": 0.1 * np.arange(10, dtype="float64") - 0.5},
            "fields": {"mass": np.ones(10, dtype="float64")},
        },
    )
    out = ds.deposit("mass", method=method, return_ghost_padded_array=True)
    assert np.sum(out) == ds.particles.count


def test_missing_particles():
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {"x": np.arange(10, dtype="float64")},
        },
    )
    with pytest.raises(
        TypeError, match="Cannot deposit particle fields on a particle-less dataset"
    ):
        ds.deposit("mass", method="ngp")


def test_missing_fields():
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {"x": np.arange(10, dtype="float64")},
        },
        particles={"coordinates": {"x": np.arange(10, dtype="float64")}},
    )
    with pytest.raises(TypeError, match="There are no particle fields"):
        ds.deposit("mass", method="ngp")


def test_unknown_field(sample_2D_dataset):
    with pytest.raises(ValueError, match="Unknown particle field 'density'"):
        sample_2D_dataset.deposit("density", method="ngp")


def test_unknown_method(sample_2D_dataset):
    with pytest.raises(
        ValueError, match="Unknown deposition method 'test', expected any of (.*)"
    ):
        sample_2D_dataset.deposit("density", method="test")


def test_callable_method(sample_2D_dataset):
    sample_2D_dataset.deposit("mass", method=_deposit_ngp_2D)


def test_callable_method_with_metadata(sample_2D_dataset):
    _is_used = False
    _md_is_received = False

    def fake_dep(*args, metadata=None, **kwargs):
        nonlocal _is_used, _md_is_received
        _is_used = True
        _md_is_received = metadata is not None
        return _deposit_ngp_2D(*args, **kwargs)

    sample_2D_dataset.deposit("mass", method=fake_dep)
    assert _is_used
    assert _md_is_received


@pytest.mark.parametrize("method", ["ngp", "cic", "tsc"])
@pytest.mark.mpl_image_compare
def test_2D_deposit(sample_2D_dataset, method):
    from matplotlib.figure import Figure

    ds = sample_2D_dataset
    particle_density = ds.deposit("mass", method=method)

    fig = Figure()
    ax = fig.add_subplot()

    im = ax.pcolormesh(
        "x",
        "y",
        particle_density.T,
        data=ds.grid.cell_edges,
        cmap="viridis",
        edgecolors="white",
    )
    ax.set(aspect=1, xlabel="x", ylabel="y", title=f"Deposition method '{method}'")
    ax.scatter(
        "x",
        "y",
        data=ds.particles.coordinates,
        edgecolor="black",
        color="tab:red",
        marker="o",
    )
    fig.colorbar(im, ax=ax)
    return fig


@pytest.mark.parametrize("method", ["ngp", "cic", "tsc"])
@pytest.mark.parametrize("grid_type", ["linear", "geometric"])
@pytest.mark.mpl_image_compare
def test_1D_deposit(method, grid_type):
    from matplotlib.figure import Figure

    if grid_type == "linear":
        xedges = np.linspace(1, 2, 6)
    elif grid_type == "geometric":
        xedges = np.geomspace(1, 2, 6)
    else:
        raise RuntimeError  # pragma: no cover

    npart = 16
    prng = np.random.RandomState(0)
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {"x": xedges},
        },
        particles={
            "coordinates": {"x": 1 + prng.random_sample(npart)},
            "fields": {"mass": np.ones(npart)},
        },
    )
    mass = ds.deposit("mass", method=method)
    if method == "ngp":
        assert mass.sum() == ds.particles.count

    fig = Figure()
    ax = fig.add_subplot()
    ax.set(xlabel="x", ylabel="particle mass", title=f"Deposition method '{method}'")
    ax.bar(
        ds.grid.cell_centers["x"], mass, width=ds.grid.cell_widths["x"], edgecolor=None
    )
    for x in ds.particles.coordinates["x"]:
        ax.axvline(x, ls="--", color="black", lw=0.4, alpha=0.6)
    return fig


@pytest.mark.parametrize("method", ["ngp", "cic", "tsc"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_3D_deposit(method, dtype):
    npart = 60
    prng = np.random.RandomState(0)
    data_2D = {
        "geometry": "cartesian",
        "grid": {
            "cell_edges": {
                "x": np.linspace(-1, 1, 10, dtype=dtype),
                "y": np.linspace(-1, 1, 10, dtype=dtype),
            },
        },
        "particles": {
            "coordinates": {
                "x": 2 * (prng.random_sample(npart).astype(dtype) - 0.5),
                "y": 2 * (prng.random_sample(npart).astype(dtype) - 0.5),
            },
            "fields": {
                "mass": np.ones(npart, dtype),
            },
        },
    }
    ds2D = gpgi.load(**data_2D)
    assert ds2D.grid.ndim == 2

    data_3D = deepcopy(data_2D)
    data_3D["grid"]["cell_edges"]["z"] = np.linspace(-1, 1, 10, dtype=dtype)
    data_3D["particles"]["coordinates"]["z"] = 2 * (
        prng.random_sample(npart).astype(dtype) - 0.5
    )
    ds3D = gpgi.load(**data_3D)
    assert ds3D.grid.ndim == 3

    # check by comparing projected 3D with direct 2D
    full_deposit_2D = ds2D.deposit(
        "mass", method=method, return_ghost_padded_array=True
    )
    full_deposit_3D = ds3D.deposit(
        "mass", method=method, return_ghost_padded_array=True
    )
    reduced_deposit = full_deposit_3D.sum(axis=2)

    rtol = 5e-7 if dtype == np.dtype("float32") else 5e-16
    npt.assert_allclose(reduced_deposit, full_deposit_2D, rtol=rtol)

    # for coverage, check that 3D deposition also works if we don't
    # return ghost layers
    deposit_3D = ds3D.deposit("mass", method=method)
    assert deposit_3D.shape == tuple(a - 2 for a in full_deposit_3D.shape)


@pytest.mark.mpl_image_compare
def test_readme_example():
    from matplotlib.figure import Figure

    nx = ny = 64
    nparticles = 600_000

    prng = np.random.RandomState(0)

    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.linspace(-1, 1, nx),
                "y": np.linspace(-1, 1, ny),
            },
        },
        particles={
            "coordinates": {
                "x": 2 * (prng.normal(0.5, 0.25, nparticles) % 1 - 0.5),
                "y": 2 * (prng.normal(0.5, 0.25, nparticles) % 1 - 0.5),
            },
            "fields": {
                "mass": np.ones(nparticles),
            },
        },
    )

    particle_mass = ds.deposit("mass", method="nearest_grid_point")

    fig = Figure()
    ax = fig.add_subplot()
    ax.set(aspect=1, xlabel="x", ylabel="y")

    im = ax.pcolormesh(
        "x",
        "y",
        particle_mass.T,
        data=ds.grid.cell_edges,
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="deposited particle mass")
    return fig


def test_performance_logging(capsys, sample_2D_dataset):
    ds = sample_2D_dataset
    ds.deposit("mass", method="ngp", verbose=True)
    stdout, stderr = capsys.readouterr()
    assert stderr == ""
    lines = stdout.strip().split("\n")
    assert len(lines) == 2
    assert re.match(r"Indexed .* particles in .* s", lines[0])
    assert re.match(r"Deposited .* particles in .* s", lines[1])


def test_return_ghost_padded_array():
    npart = 16
    prng = np.random.RandomState(0)
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {"x": np.linspace(1, 2, 6)},
        },
        particles={
            "coordinates": {"x": 1 + prng.random_sample(npart)},
            "fields": {"mass": np.ones(npart)},
        },
    )
    active_array = ds.deposit("mass", method="cic")
    assert active_array.shape == (5,)
    padded_array = ds.deposit(
        "mass", method="cic", verbose=True, return_ghost_padded_array=True
    )
    assert padded_array.shape == (7,)


@pytest.mark.parametrize(
    "boundaries, error_type, error_message",
    [
        (
            {"radius": ("open", "open")},
            ValueError,
            "Got invalid ax key 'radius', expected any of ('x', 'y')",
        ),
        (
            {"x": ("o", "o", "o")},
            TypeError,
            "Expected a 2-tuple of strings, got ('o', 'o', 'o')",
        ),
        (
            {"x": ("open", "unregistered_key")},
            ValueError,
            "Unknown boundary type 'unregistered_key'",
        ),
    ],
)
def test_deposit_invalid_boundaries(
    boundaries, error_type, error_message, sample_2D_dataset
):
    ds = sample_2D_dataset
    with pytest.raises(error_type, match=re.escape(error_message)):
        ds.deposit("mass", method="ngp", boundaries=boundaries)


@pytest.mark.parametrize(
    "keyL, keyR",
    [("open", "wall"), ("periodic", "periodic"), ("antisymmetric ", "open")],
)
def test_builtin_boundary_recipe(keyL, keyR, sample_2D_dataset):
    ds = sample_2D_dataset
    res1 = ds.deposit("mass", method="tsc", boundaries={"x": (keyL, keyR)})

    if keyR == keyL:
        return
    res2 = ds.deposit("mass", method="tsc", boundaries={"x": (keyR, keyL)})
    npt.assert_array_equal(res1[1:-1, 1:-1], res2[1:-1, 1:-1])


def test_register_invalid_boundary_recipe():
    nx = ny = 64
    nparticles = 100

    prng = np.random.RandomState(0)
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.linspace(-1, 1, nx),
                "y": np.linspace(-1, 1, ny),
            },
        },
        particles={
            "coordinates": {
                "x": 2 * (prng.normal(0.5, 0.25, nparticles) % 1 - 0.5),
                "y": 2 * (prng.normal(0.5, 0.25, nparticles) % 1 - 0.5),
            },
            "fields": {
                "mass": np.ones(nparticles),
            },
        },
    )

    def _my_recipe(a, b, c, d, e, f):
        return a  # pragma: no cover

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid boundary recipe. Expected a function with exactly 6 parameters, "
            "named 'same_side_active_layer', 'same_side_ghost_layer', "
            "'opposite_side_active_layer', 'opposite_side_ghost_layer', "
            "'weight_same_side_active_layer', 'weight_same_side_ghost_layer', "
            "'weight_opposite_side_active_layer', 'weight_opposite_side_ghost_layer', "
            "'side', and 'metadata'"
        ),
    ):
        ds.boundary_recipes.register("my", _my_recipe)


def test_register_custom_boundary_recipe(sample_2D_dataset):
    def _my_recipe(
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
    ):
        # return the active layer unchanged
        # (this is the same as the builtin 'open' boundary recipe)
        return same_side_active_layer

    ds = sample_2D_dataset
    ds.boundary_recipes.register("my", _my_recipe)
    res1 = ds.deposit("mass", method="tsc", boundaries={"x": ("my", "my")})
    res2 = ds.deposit("mass", method="tsc")
    npt.assert_array_equal(res1, res2)


@pytest.fixture
def unary_mass_dataset():
    from itertools import product

    x = np.linspace(0.5, 9.5, 10)
    xc, yc, zc = np.array(list(product(x, x, x))).T

    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.linspace(0, 10, 11),
                "y": np.linspace(0, 10, 11),
                "z": np.linspace(0, 10, 11),
            },
        },
        particles={
            "coordinates": {
                "x": xc,
                "y": yc,
                "z": zc,
            },
            "fields": {"mass": 1e-3 * np.ones(1000)},
        },
    )
    return ds


@pytest.mark.parametrize("method", ["ngp", "cic", "tsc"])
def test_unary_mass_dataset(method, unary_mass_dataset):
    # check that total mass is conserved with open boundaries
    # when including ghost zones
    ds = unary_mass_dataset
    base_array = ds.deposit("mass", method=method, return_ghost_padded_array=True)
    expected = ds.particles.fields["mass"].sum()
    assert base_array.sum() == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "boundaries",
    [
        {
            "x": ("periodic", "periodic"),
            "y": ("periodic", "periodic"),
            "z": ("periodic", "periodic"),
        },
        {
            "x": ("wall", "wall"),
            "y": ("wall", "wall"),
            "z": ("wall", "wall"),
        },
    ],
)
@pytest.mark.parametrize("method", ["ngp", "cic", "tsc"])
def test_closed_boundaries(method, boundaries, unary_mass_dataset):
    ds = unary_mass_dataset
    arr = ds.deposit("mass", method=method, boundaries=boundaries)
    expected = ds.particles.fields["mass"].sum()
    assert arr.sum() == pytest.approx(expected, rel=1e-12)


@pytest.mark.skipif(
    gpgi._IS_PY_LIB,
    reason="weight_field is not implemented with gpgi._IS_PY_LIB = True",
)
def test_deposit_with_weight_field(sample_2D_dataset):
    sample_2D_dataset.deposit("vx", method="tsc", weight_field="mass")


def test_deposit_with_weight_field_and_incomplete_boundaries_spec(sample_2D_dataset):
    with pytest.raises(
        TypeError,
        match="weight_field_boundaries keyword argument is required with weight_field and boundaries",
    ):
        sample_2D_dataset.deposit(
            "vx",
            method="tsc",
            weight_field="mass",
            boundaries={"x": ("periodic", "periodic")},
        )


def test_deposit_with_weight_field_and_complete_boundaries_spec(sample_2D_dataset):
    ds = sample_2D_dataset
    ds.deposit(
        "vx",
        method="ngp",
        weight_field="mass",
        boundaries={"x": ("periodic", "periodic")},
        weight_field_boundaries={"x": ("periodic", "periodic")},
    )


def test_warn_unused_weight_field_boundaries(sample_2D_dataset):
    with pytest.warns(
        UserWarning,
        match=(
            "weight_field_boundaries will not be used as no weight_field was specified"
        ),
    ):
        sample_2D_dataset.deposit(
            "vx",
            method="ngp",
            weight_field_boundaries={"x": ("periodic", "periodic")},
        )


def test_partial_boundary():
    def _base_recipe(
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
        *,
        mode,  # supplementary argument that is destined to be frozen with functools.partial
    ):
        # the return value doesn't matter, this test checks that
        # recipe validation doesn't fail
        return same_side_active_layer  # pragma: no cover

    myrecipe = partial(_base_recipe, mode="test")

    # define a minimal single-use dataset
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.linspace(0, 10, 11),
                "y": np.linspace(0, 10, 11),
                "z": np.linspace(0, 10, 11),
            },
        },
    )
    ds.boundary_recipes.register("my", myrecipe, skip_validation=False)


@pytest.mark.parametrize("method", ["ngp", "cic", "tsc"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_particles_on_domain_corners(method, dtype):
    nx, ny, nz = 2, 3, 5
    xlim = ylim = zlim = (0, 1)
    nparticles = 6
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.linspace(*xlim, nx + 1, dtype=dtype),
                "y": np.linspace(*ylim, ny + 1, dtype=dtype),
                "z": np.linspace(*zlim, nz + 1, dtype=dtype),
            },
        },
        particles={
            "coordinates": {
                "x": np.array([0, 0, 0, 1, 1, 1], dtype=dtype),
                "y": np.array([0, 1, 1, 0, 1, 1], dtype=dtype),
                "z": np.array([1, 0, 1, 0, 0, 1], dtype=dtype),
            },
            "fields": {
                "mass": np.ones(nparticles, dtype=dtype),
            },
        },
    )
    ds.deposit("mass", method=method)


@pytest.mark.parametrize("lock", ["per-instance", None, Lock()])
def test_explicit_lock(lock, sample_2D_dataset):
    ds = sample_2D_dataset
    mass_dep_ref = ds.deposit("mass", method="nearest_grid_point")
    mass_dep_exp = ds.deposit("mass", method="nearest_grid_point", lock=lock)
    npt.assert_array_equal(mass_dep_exp, mass_dep_ref)


@pytest.mark.parametrize("lock", ["perinstance", 0, nullcontext()])
def test_invalick_lock(lock, sample_2D_dataset):
    ds = sample_2D_dataset
    with pytest.raises(ValueError, match="Received lock="):
        ds.deposit("mass", method="nearest_grid_point", lock=lock)
