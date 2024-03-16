"""gpgi: Fast particle deposition at post-processing time."""

from typing import Any

from .types import Dataset, FieldMap, Geometry, Grid, ParticleSet

__version__ = "1.0.0"


def load(
    *,
    geometry: str = "cartesian",
    grid: dict[str, FieldMap] | None = None,
    particles: dict[str, FieldMap] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Dataset:
    r"""
    Load a Dataset.

    Parameters
    ----------
    geometry: Literal["cartesian", "polar", "cylindrical", "spherical", "equatorial"]
        This flag is used for validation of axis names, order and domain limits.

    grid: dict[str, FieldMap] (optional)
        A dictionary representing the grid coordinates as 1D arrays of cell left edges,
        and on-grid fields as ND arrays (fields are assumed to be defined on cell
        centers)

    particles: dict[str, FieldMap] (optional)
        A dictionary representing particle coordinates and associated fields as 1D
        arrays

    metadata: dict[str, Any] (optional)
        A dictionnary representing arbitrary additional data, that will be attached to
        the returned Dataset as an attribute (namely, ds.metadata). This special
        attribute is accessible from boundary condition methods as the argument of the
        same name.
    """
    try:
        _geometry = Geometry(geometry)
    except ValueError:
        raise ValueError(
            f"unknown geometry {geometry!r}, expected any of {tuple(_.value for _ in Geometry)}"
        ) from None

    _grid: Grid | None = None
    if grid is not None:
        if "cell_edges" not in grid:
            raise ValueError("grid dictionary missing required key 'cell_edges'")
        _grid = Grid(
            _geometry, cell_edges=grid["cell_edges"], fields=grid.get("fields", {})
        )

    _particles: ParticleSet | None = None
    if particles is not None:
        if "coordinates" not in particles:
            raise ValueError("particles dictionary missing required key 'coordinates'")
        _particles = ParticleSet(
            _geometry,
            coordinates=particles["coordinates"],
            fields=particles.get("fields", {}),
        )

    return Dataset(
        geometry=_geometry, grid=_grid, particles=_particles, metadata=metadata
    )
