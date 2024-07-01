from textwrap import dedent

import numpy as np

import gpgi


def test_repr():
    ds = gpgi.load(
        geometry="cartesian",
        grid={"cell_edges": {"x": np.array([0.0, 1.0])}},
        particles={"coordinates": {"x": np.array([0.5])}},
    )

    expected_grid_repr = dedent(
        """\
        Grid(
            geometry='cartesian',
            cell_edges={'x': array([0., 1.])},
            fields={},
        )"""
    )
    assert repr(ds.grid) == expected_grid_repr

    expected_particles_repr = dedent(
        """\
        ParticleSet(
            geometry='cartesian',
            coordinates={'x': array([0.5])},
            fields={},
        )"""
    )
    assert repr(ds.particles) == expected_particles_repr

    expected_ds_repr = dedent(
        """\
        Dataset(
            geometry='cartesian',
            grid=Grid(
                geometry='cartesian',
                cell_edges={'x': array([0., 1.])},
                fields={},
            ),
            particles=ParticleSet(
                geometry='cartesian',
                coordinates={'x': array([0.5])},
                fields={},
            ),
            metadata={},
        )"""
    )
    assert repr(ds) == expected_ds_repr
