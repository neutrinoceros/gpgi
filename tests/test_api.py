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
