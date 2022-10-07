from __future__ import annotations

from .types import Dataset
from .types import FieldMap
from .types import Geometry
from .types import Grid
from .types import ParticleSet

_geometry_names: dict[str, Geometry] = {g.name.lower(): g for g in Geometry}


def load(
    *,
    geometry: str = "cartesian",
    grid: dict[str, FieldMap] | None = None,
    particles: dict[str, FieldMap] | None = None,
) -> Dataset:

    if geometry in _geometry_names:
        _geometry = _geometry_names[geometry]
    else:
        raise ValueError(
            f"unknown geometry {geometry!r}, expected any of {tuple(_geometry_names.keys())}"
        )

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

    return Dataset(geometry=_geometry, grid=_grid, particles=_particles)
