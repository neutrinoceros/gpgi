from __future__ import annotations

from typing import TYPE_CHECKING, cast

from gpgi._data_types import Dataset, Geometry, Grid, ParticleSet

if TYPE_CHECKING:
    from typing import Any, Literal

    from gpgi._typing import FieldMap, FloatT, GridDict, ParticleSetDict


def load(
    *,
    geometry: Literal["cartesian", "polar", "cylindrical", "spherical", "equatorial"],
    grid: GridDict[FloatT],
    particles: ParticleSetDict[FloatT] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Dataset[FloatT]:
    r"""
    Load a Dataset.

    Parameters
    ----------
    geometry: Literal["cartesian", "polar", "cylindrical", "spherical", "equatorial"]
        This flag is used for validation of axis names, order and domain limits.

    grid: dict[str, FieldMap]
        A dictionary representing the grid coordinates as 1D arrays of cell left edges,
        and on-grid fields as ND arrays (fields are assumed to be defined on cell
        centers)

    particles: dict[str, FieldMap] (optional)
        A dictionary representing particle coordinates and associated fields as 1D
        arrays

    metadata: dict[str, Any] (optional)
        A dictionary representing arbitrary additional data, that will be attached to
        the returned Dataset as an attribute (namely, ds.metadata). This special
        attribute is accessible from boundary condition methods as the argument of the
        same name.
    """
    try:
        # when Python 3.11 is dropped, this try-except can be simplified as
        # if geometry not in Geometry: raise
        _geometry = Geometry(geometry)
    except ValueError:
        # this exception is raised directly, not added to a stack,
        # in order to avoid "possibly unbound" warnings from static analysis
        raise ValueError(
            f"unknown geometry {geometry!r}, "
            f"expected any of {tuple(_.value for _ in Geometry)}"
        ) from None

    exceptions: list[Exception] = []

    if "cell_edges" not in grid:
        exceptions.append(
            ValueError("grid dictionary missing required key 'cell_edges'")
        )
    if particles is not None and "coordinates" not in particles:
        exceptions.append(
            ValueError("particles dictionary missing required key 'coordinates'")
        )
    if len(exceptions) == 1:
        raise exceptions[0]
    elif exceptions:
        raise ExceptionGroup("Invalid inputs were received", exceptions)

    _grid = Grid(
        geometry=_geometry,
        cell_edges=cast("FieldMap[FloatT]", grid["cell_edges"]),
        fields=grid.get("fields"),
    )

    _particles: ParticleSet[FloatT] | None = None
    if particles is not None:
        _particles = ParticleSet(
            geometry=_geometry,
            coordinates=cast("FieldMap[FloatT]", particles["coordinates"]),
            fields=particles.get("fields"),
        )

    return Dataset(
        geometry=_geometry, grid=_grid, particles=_particles, metadata=metadata
    )
