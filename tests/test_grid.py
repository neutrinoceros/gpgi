import numpy as np
import numpy.testing as npt
import pytest

import gpgi


def test_cell_volumes_cartesian():
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.arange(11, dtype="float64"),
                "y": np.arange(0, 1.1, 0.1, dtype="float64"),
            }
        },
    )
    expected = np.prod(np.meshgrid(np.ones(10), 0.1 * np.ones(10)), axis=0)
    assert ds.grid.cell_volumes.dtype == expected.dtype
    npt.assert_allclose(ds.grid.cell_volumes, expected, rtol=1e-15)


def test_cell_volumes_curvilinear():
    ds = gpgi.load(
        geometry="cylindrical",
        grid={
            "cell_edges": {
                "radius": np.arange(11, dtype="float64"),
                "z": np.arange(0, 1.1, 0.1),
            }
        },
    )
    with pytest.raises(
        NotImplementedError,
        match=r"cell_volumes property is not implemented for cylindrical geometry",
    ):
        ds.grid.cell_volumes  # noqa: B018


def test_cell_volumes_shape():
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.linspace(0, 1, 3),
                "y": np.linspace(0, 1, 4),
                "z": np.linspace(0, 1, 5),
            }
        },
    )
    assert ds.grid.cell_volumes.shape == ds.grid.shape


def test_cell_centers():
    ds = gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.linspace(0, 1, 3),
                "y": np.linspace(0, 1, 4),
                "z": np.linspace(0, 1, 5),
            }
        },
    )
    for key in "xyz":
        assert ds.grid.cell_centers[key].min() > ds.grid.cell_edges[key].min()
        assert ds.grid.cell_centers[key].max() < ds.grid.cell_edges[key].max()
