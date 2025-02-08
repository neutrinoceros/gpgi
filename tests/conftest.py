import sys
from importlib.metadata import version
from importlib.util import find_spec

import pytest

import gpgi

HAVE_PYTEST_MPL = find_spec("pytest_mpl") is not None


def pytest_configure(config):
    if HAVE_PYTEST_MPL:
        import matplotlib as mpl

        mpl.use("Agg")
    else:  # pragma: no cover
        config.addinivalue_line(
            "markers", "mpl_image_compare: skip (missing requirement: pytest_mpl)"
        )


def pytest_runtest_setup(item):
    if HAVE_PYTEST_MPL:
        return
    if any(item.iter_markers(name="mpl_image_compare")):  # pragma: no cover
        pytest.skip("missing requirement: pytest_mpl")


def pytest_report_header(config, start_path):
    is_gil_enabled = sys.version_info < (3, 13) or sys._is_gil_enabled()

    return [
        f"{is_gil_enabled = }",
        f"NumPy: {version('numpy')}",
        f"{gpgi._IS_PY_LIB = }",
        f"gpgi._lib loads from {find_spec('gpgi._lib').origin}",
    ]
