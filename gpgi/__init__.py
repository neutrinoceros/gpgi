from __future__ import annotations

import enum
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import numpy.typing as npt


class Geometry(enum.Enum):
    CARTESIAN = enum.auto()
    POLAR = enum.auto()
    CYLINDRICAL = enum.auto()
    SPHERICAL = enum.auto()


_geometry_names: dict[str, Geometry] = {
    "cartesian": Geometry.CARTESIAN,
    "polar": Geometry.POLAR,
    "cylindrical": Geometry.CYLINDRICAL,
    "spherical": Geometry.SPHERICAL,
}


class DepositionMethod(enum.Enum):
    PARTICLE_IN_CELL = enum.auto()
    CLOUD_IN_CELL = enum.auto()
    TRIANGULAR_SHAPED_CLOUD = enum.auto()


_deposition_method_names: dict[str, DepositionMethod] = {
    "pic": DepositionMethod.PARTICLE_IN_CELL,
    "cic": DepositionMethod.CLOUD_IN_CELL,
    "tsc": DepositionMethod.TRIANGULAR_SHAPED_CLOUD,
}

Name = str
FieldMap = dict[Name, np.ndarray]


class ValidatorMixin(ABC):
    @abstractmethod
    def validate(self) -> None:
        ...

    def _validate_fieldmaps(
        self,
        *fmaps: FieldMap | None,
        require_shape_equality: bool = True,
        requirements: dict[Name, Any] | None = None,
    ):
        if not fmaps:
            return

        _reference_shape: tuple[int, ...] | None = None
        _reference_field_name: str
        for fmap in fmaps:
            if fmap is None:
                continue
            for name, data in fmap.items():
                if require_shape_equality:
                    if _reference_shape is None:
                        _reference_shape = data.shape
                        _reference_field_name = name
                    elif data.shape != _reference_shape:
                        raise ValueError(
                            f"Fields {name!r} and {_reference_field_name!r} "
                            f"have mismatching shapes {data.shape} and {_reference_shape}"
                        )

                # optional size checks by order of increasing strictness
                if not requirements:
                    continue
                for attr, expected in requirements.items():
                    if (actual := getattr(data, attr)) != expected:
                        raise ValueError(
                            f"Field {name!r} has incorrect {attr} {actual} "
                            f"(expected {expected})"
                        )

    def _validate_geometry(self):

        known_axes: tuple[Name, Name, Name] = {
            Geometry.CARTESIAN: ("x", "y", "z"),
            Geometry.POLAR: ("radius", "z", "azimuth"),
            Geometry.CYLINDRICAL: ("radius", "azimuth", "z"),
            Geometry.SPHERICAL: ("radius", "colatitude", "azimuth"),
        }
        if self.geometry not in known_axes:
            raise ValueError(f"Unknown geometry {self.geometry.name.lower()!r}")

        axes = known_axes[self.geometry]
        for expected, actual in zip(axes, self.axes):
            if actual != expected:
                raise ValueError(
                    f"Got invalid axis {actual!r} with geometry {self.geometry.name.lower()!r}"
                )


@dataclass
class Grid(ValidatorMixin):
    geometry: Geometry
    cell_edges: FieldMap
    fields: FieldMap | None

    def validate(self) -> None:
        self._validate_geometry()
        self._validate_fieldmaps(
            self.cell_edges, require_shape_equality=False, requirements={"ndim": 1}
        )
        self._validate_fieldmaps(
            self.fields,
            requirements={
                "size": self.size,
                "ndim": self.ndim,
                "shape": self.shape,
            },
        )

    @property
    def axes(self) -> tuple[Name, ...]:
        return tuple(self.cell_edges.keys())

    @cached_property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(_) - 1 for _ in self.cell_edges.values())

    @property
    def size(self):
        return np.product(self.shape)

    @property
    def ndim(self) -> int:
        return len(self.axes)


@dataclass
class ParticleSet(ValidatorMixin):
    geometry: Geometry
    positions: FieldMap
    fields: FieldMap | None

    def validate(self) -> None:
        self._validate_geometry()
        self._validate_fieldmaps(self.positions, self.fields, requirements={"ndim": 1})

    @property
    def axes(self) -> tuple[Name, ...]:
        return tuple(self.positions.keys())

    @property
    def count(self):
        for p in self.positions.values():
            return len(p)

    @property
    def ndim(self) -> int:
        return len(self.axes)


def _deposit_pic(pcount, hci, pfield, buffer):
    for ipart in range(pcount):
        md_idx = tuple(hci[ipart])
        buffer[md_idx] += pfield[ipart].d


@dataclass
class Dataset:
    geometry: Geometry = Geometry.CARTESIAN
    grid: Grid | None = None
    particles: ParticleSet | None = None

    def _setup_host_cell_index(self):
        if hasattr(self, "_hci"):
            return
        self._hci = np.empty((self.particles.count, self.grid.ndim), dtype="int64")
        for ipart in range(self.particles.count):
            for idim, ax in enumerate(self.grid.axes):
                idx = 0
                x = self.particles.positions[ax][ipart]
                edges = self.grid.cell_edges[ax]
                max_idx = len(edges) - 1
                while idx <= max_idx and edges[idx + 1] < x:
                    idx += 1
                self._hci[ipart, idim] = idx

    def deposit(self, particle_field_key: Name, /, *, method: Name) -> np.ndarray:
        # public interface
        if method in _deposition_method_names:
            met = _deposition_method_names[method]
        else:
            raise ValueError(
                f"unknown deposition method {method!r}, "
                f"expected any of {tuple(_deposition_method_names.keys())}"
            )
        if self.grid is None:
            raise TypeError("Cannot deposit particle fields on a grid-less dataset")
        if self.particles is None:
            raise TypeError("Cannot deposit particle fields on a particle-less dataset")
        if self.particles.fields is None:
            raise TypeError("There are no particle fields")
        if particle_field_key not in self.particles.fields:
            raise ValueError(f"Unknown particle field {particle_field_key!r}")

        # deactivating type checking for deposition methods because
        # they may be ported to Cyhton later
        known_methods: dict[DepositionMethod, Any] = {
            DepositionMethod.PARTICLE_IN_CELL: _deposit_pic,
        }
        if met not in known_methods:
            raise NotImplementedError(f"method {method} is not implemented yet")

        pfield = self.particles.fields[particle_field_key]
        ret_array = np.zeros(self.grid.shape)
        self._setup_host_cell_index()

        known_methods[met](self.particles.count, self._hci, pfield, ret_array)
        return ret_array


# public interface
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
            _geometry, cell_edges=grid["cell_edges"], fields=grid.get("fields")
        )
        _grid.validate()

    _particles: ParticleSet | None = None
    if particles is not None:
        if "positions" not in particles:
            raise ValueError("particles dictionary missing required key 'positions'")
        _particles = ParticleSet(
            _geometry, positions=particles["positions"], fields=particles.get("fields")
        )
        _particles.validate()

    return Dataset(geometry=_geometry, grid=_grid, particles=_particles)
