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
            "coordinates": {
                "x": (2 * prng.random_sample(nparticles) - 1) * un.m,
                "y": (2 * prng.random_sample(nparticles)) * un.m,
            },
            "fields": {
                "mass": prng.random_sample(nparticles) * un.g,
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
    ds = sample_dataset
    particle_density = ds.deposit("mass", method="pic")

    # a second call should yield the same exact array
    particle_density_2 = ds.deposit("mass", method="pic")
    assert particle_density_2 is particle_density


@pytest.mark.mpl_image_compare
def test_deposit_image(sample_dataset):
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
                data=ds.particles.coordinates,
                edgecolor="black",
                color="tab:red",
                marker="o",
            )
        fig.colorbar(im, ax=ax)
    return fig
