from importlib.util import find_spec

import pytest

HAVE_PYTEST_MPL = find_spec("pytest_mpl") is not None


def pytest_configure(config):
    if not HAVE_PYTEST_MPL:  # pragma: no cover
        config.addinivalue_line(
            "markers", "mpl_image_compare: skip (missing requirement: pytest_mpl)"
        )


def pytest_runtest_setup(item):
    if HAVE_PYTEST_MPL:
        return
    if any(item.iter_markers(name="mpl_image_compare")):  # pragma: no cover
        pytest.skip("missing requirement: pytest_mpl")


def pytest_report_header(config, start_path):
    return f"gpgi._lib loads from {find_spec('gpgi._lib').origin}"
