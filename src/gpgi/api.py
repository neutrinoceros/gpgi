from __future__ import annotations

from typing import Any

from .types import Dataset, FieldMap, Geometry, Grid, ParticleSet


def load(
    *,
    geometry: str = "cartesian",
    grid: dict[str, FieldMap] | None = None,
    particles: dict[str, FieldMap] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Dataset:
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
