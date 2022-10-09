import re

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
import unyt as un

import gpgi


@pytest.fixture()
def sample_dataset():
    nparticles = 100
    nx, ny = grid_shape = 16, 16
    prng = np.random.RandomState(0)

    return gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.linspace(-1, 1, nx + 1) * un.m,
                "y": np.linspace(0, 2, ny + 1) * un.m,
            },
            "fields": {
                "density": np.ones(grid_shape) * un.g / un.cm**3,
            },
        },
        particles={
            "coordinates": {
                "x": (2 * prng.random_sample(nparticles) - 1) * un.m,
                "y": (2 * prng.random_sample(nparticles)) * un.m,
            },
            "fields": {
                "mass": np.ones(nparticles) * un.g,
            },
        },
    )


def test_missing_grid():
    ds = gpgi.load(
        geometry="cartesian",
        particles={
            "coordinates": {"x": np.arange(10)},
            "fields": {"mass": np.ones(10)},
        },
    )
    with pytest.raises(
        TypeError, match="Cannot deposit particle fields on a grid-less dataset"
    ):
        ds.deposit("mass", method="pic")


def test_missing_particles():
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {"x": np.arange(10)},
        },
    )
    with pytest.raises(
        TypeError, match="Cannot deposit particle fields on a particle-less dataset"
    ):
        ds.deposit("mass", method="pic")


def test_missing_fields():
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {"x": np.arange(10)},
        },
        particles={"coordinates": {"x": np.arange(10)}},
    )
    with pytest.raises(TypeError, match="There are no particle fields"):
        ds.deposit("mass", method="pic")


def test_unknown_field(sample_dataset):
    with pytest.raises(ValueError, match="Unknown particle field 'density'"):
        sample_dataset.deposit("density", method="pic")


def test_unknown_method(sample_dataset):
    with pytest.raises(
        ValueError, match="Unknown deposition method 'test', expected any of (.*)"
    ):
        sample_dataset.deposit("density", method="test")


def test_double_deposit(sample_dataset):
    # this test intentionally dwelves into the realm of
    # checking implementation details rather than just checking behaviour
    ds = sample_dataset
    assert len(ds._cache) == 0

    particle_density = ds.deposit("mass", method="pic")
    assert len(ds._cache) == 1

    # a second call should yield the same exact array
    particle_density_2 = ds.deposit("mass", method="pic")
    npt.assert_array_equal(particle_density_2, particle_density)
    assert particle_density_2.base is particle_density.base
    assert len(ds._cache) == 1

    # using the full key shouldn't produce another array
    particle_density_3 = ds.deposit("mass", method="particle_in_cell")
    npt.assert_array_equal(particle_density_3, particle_density)
    assert particle_density_3.base is particle_density.base
    assert len(ds._cache) == 1


@pytest.mark.parametrize("method", ["pic", "cic", "tsc"])
@pytest.mark.mpl_image_compare
def test_2D_deposit(sample_dataset, method):
    ds = sample_dataset
    particle_density = ds.deposit("mass", method=method)

    fig, ax = plt.subplots()

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


@pytest.mark.parametrize("method", ["pic", "cic", "tsc"])
@pytest.mark.parametrize("grid_type", ["linear", "geometric"])
@pytest.mark.mpl_image_compare
def test_1D_deposit(method, grid_type):

    xedges = {"linear": np.linspace(1, 2, 6), "geometric": np.geomspace(1, 2, 6)}[
        grid_type
    ]

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
    if method == "pic":
        assert mass.sum() == ds.particles.count

    fig, ax = plt.subplots()
    ax.set(xlabel="x", ylabel="particle mass", title=f"Deposition method '{method}'")
    ax.bar(
        ds.grid.cell_centers["x"], mass, width=ds.grid.cell_widths["x"], edgecolor=None
    )
    for x in ds.particles.coordinates["x"]:
        ax.axvline(x, ls="--", color="black", lw=0.4, alpha=0.6)
    return fig


@pytest.mark.parametrize("method", ["pic", "cic", "tsc"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_3D_deposit(method, dtype):
    npart = 60
    prng = np.random.RandomState(0)
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.linspace(-1, 1, 10, dtype=dtype),
                "y": np.linspace(-1, 1, 10, dtype=dtype),
                "z": np.linspace(-1, 1, 10, dtype=dtype),
            },
        },
        particles={
            "coordinates": {
                "x": 2 * (prng.random_sample(npart).astype(dtype) - 0.5),
                "y": 2 * (prng.random_sample(npart).astype(dtype) - 0.5),
                "z": 2 * (prng.random_sample(npart).astype(dtype) - 0.5),
            },
            "fields": {
                "mass": np.ones(npart, dtype),
            },
        },
    )
    ds.deposit("mass", method=method)


@pytest.mark.mpl_image_compare
def test_readme_example():
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

    particle_mass = ds.deposit("mass", method="particle_in_cell")

    fig, ax = plt.subplots()
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


def test_performance_logging(capsys):
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
    ds.deposit("mass", method="pic", verbose=True)
    stdout, stderr = capsys.readouterr()
    assert stderr == ""
    lines = stdout.strip().split("\n")
    assert len(lines) == 2
    assert re.match(r"Indexed .* particles in .* s", lines[0])
    assert re.match(r"Deposited .* particles in .* s", lines[1])


def test_return_ghost_padded_array(capsys):

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
    assert active_array.base is padded_array

    # check that no expensive operation is logged:
    # the whole padded array should be cached after the first deposition
    stdout, stderr = capsys.readouterr()
    assert stdout == ""
    assert stderr == ""
