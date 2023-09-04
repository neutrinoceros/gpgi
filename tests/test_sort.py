import re

import numpy as np
import pytest

import gpgi


def get_random_dataset(dimensionality: int):
    NPARTICLES = 10_000
    NX = 16
    prng = np.random.RandomState(0)

    cell_edges = {
        "x": np.linspace(-1, 1, NX + 1),
    }
    coordinates = {"x": (2 * prng.random_sample(NPARTICLES) - 1)}
    if dimensionality >= 2:
        cell_edges["y"] = np.linspace(-1, 1, NX + 1)
        coordinates["y"] = 2 * prng.random_sample(NPARTICLES) - 1
    if dimensionality == 3:
        cell_edges["z"] = np.linspace(-1, 1, NX + 1)
        coordinates["z"] = 2 * prng.random_sample(NPARTICLES) - 1

    return gpgi.load(
        geometry="cartesian",
        grid={"cell_edges": cell_edges},
        particles={"coordinates": coordinates},
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_sort(dim):
    ds = get_random_dataset(dim)
    assert not ds.is_sorted()
    sds = ds.sorted()
    assert sds.is_sorted()


@pytest.mark.parametrize(
    "axes, expected_message",
    [
        ((0, 1), "Expected exactly 3 axes, got 2"),
        (
            (0.0, 1.0, 2.0),
            (
                "Expected all axes to be integers, got (0.0, 1.0, 2.0) "
                "with types (float, float, float)"
            ),
        ),
        ((1, 2, 3), "Expected all axes to be <3, got (1, 2, 3)"),
    ],
)
def test_key_errors(axes, expected_message):
    ds = get_random_dataset(3)
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        ds.sorted(axes=axes)
