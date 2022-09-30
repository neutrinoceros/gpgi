import re

import matplotlib.pyplot as plt
import numpy as np
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
            "positions": {
                "x": (2 * prng.random_sample(nparticles) - 1) * un.m,
                "y": (2 * prng.random_sample(nparticles)) * un.m,
            },
            "fields": {
                "mass": prng.random_sample(nparticles) * un.g,
            },
        },
    )


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
            "positions": {
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


def test_load_invalid_grid():
    with pytest.raises(
        ValueError, match="grid dictionary missing required key 'cell_edges'"
    ):
        gpgi.load(geometry="cartesian", grid={})


def test_load_invalid_particles():
    with pytest.raises(
        ValueError, match="particles dictionary missing required key 'positions'"
    ):
        gpgi.load(geometry="cartesian", particles={})


def test_inconsistent_shape_particle_data():
    with pytest.raises(
        ValueError,
        match=re.escape(r"Fields 'y' and 'x' have mismatching shapes (3,) and (2,)"),
    ):
        gpgi.load(
            geometry="cartesian",
            particles={
                "positions": {
                    "x": np.array([0, 0]),
                    "y": np.array([0, 0, 0]),
                }
            },
        )


def test_inconsistent_dim_grid_data():
    with pytest.raises(
        ValueError,
        match=re.escape(r"Field 'density' has incorrect dimensionality 1 (expected 2)"),
    ):
        gpgi.load(
            geometry="cartesian",
            grid={
                "cell_edges": {
                    "x": np.array([0, 0]),
                    "y": np.array([0, 0, 0]),
                },
                "field": {
                    "density": np.ones(6),
                    "velocity_x": np.ones(5),
                },
            },
        )


def test_inconsistent_sizes_grid_data():
    with pytest.raises(
        ValueError,
        match=re.escape(r"Field 'velocity_x' has incorrect size 5 (expected 6)"),
    ):
        gpgi.load(
            geometry="cartesian",
            grid={
                "cell_edges": {
                    "x": np.array([0, 0]),
                    "y": np.array([0, 0, 0]),
                },
                "field": {
                    "density": np.ones(6),
                    "velocity_x": np.ones(5),
                },
            },
        )


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


@pytest.mark.mpl_image_compare
def test_particle_deposition(sample_dataset):
    ds = sample_dataset
    gas_density = ds.grid.fields["density"]
    particle_density = ds.deposit("mass", method="pic")

    mass_ratio = particle_density / gas_density

    fig, axs = plt.subplots(ncols=2, figsize=(13, 4.75), sharex=True, sharey=True)
    for field, ax in zip((gas_density, particle_density, mass_ratio), axs):
        field = field.T
        im = ax.pcolormesh(
            "x",
            "y",
            field,
            data=ds.grid.cell_edges,
            cmap="viridis",
        )
        ax.set(aspect=1, xlabel="x")
        if ax is axs[0]:
            ax.set_ylabel("y")
            ax.scatter(
                "x",
                "y",
                data=ds.particles.positions,
                edgecolor="black",
                color="tab:red",
                marker="o",
            )
        fig.colorbar(im, ax=ax)
    return fig
