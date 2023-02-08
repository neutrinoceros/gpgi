import numpy as np
import numpy.testing as npt
import pytest

from gpgi.api import load


def test_cell_volumes_cartesian():
    ds = load(
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
    ds = load(
        geometry="cylindrical",
        grid={
            "cell_edges": {
                "radius": np.arange(11, dtype="int64"),
                "z": np.arange(0, 1.1, 0.1),
            }
        },
    )
    with pytest.raises(
        NotImplementedError,
        match=r"cell_volumes property is not implemented for cylindrical geometry",
    ):
        ds.grid.cell_volumes
