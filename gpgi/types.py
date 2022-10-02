from __future__ import annotations

import enum
from abc import ABC
from abc import abstractmethod
from functools import cached_property
from functools import reduce
from typing import Any
from typing import Protocol

import numpy as np


class Geometry(enum.Enum):
    CARTESIAN = enum.auto()
    POLAR = enum.auto()
    CYLINDRICAL = enum.auto()
    SPHERICAL = enum.auto()


class DepositionMethod(enum.Enum):
    PARTICLE_IN_CELL = enum.auto()
    CLOUD_IN_CELL = enum.auto()
    TRIANGULAR_SHAPED_CLOUD = enum.auto()


_deposition_method_names: dict[str, DepositionMethod] = {
    m.name.lower(): m for m in DepositionMethod
} | {"".join([w[0] for w in m.name.split("_")]).lower(): m for m in DepositionMethod}


Name = str
FieldMap = dict[Name, np.ndarray]


class GeometricData(Protocol):
    geometry: Geometry
    axes: tuple[Name, ...]


class CoordinateData(Protocol):
    geometry: Geometry
    axes: tuple[Name, ...]
    coordinates: FieldMap


class ValidatorMixin(GeometricData, ABC):
    def __init__(self) -> None:
        self._validate()

    @abstractmethod
    def _validate(self) -> None:
        ...

    def _validate_fieldmaps(
        self,
        *fmaps: FieldMap | None,
        require_shape_equality: bool = False,
        require_sorted: bool = False,
        **required_attrs: Any,
    ) -> None:
        _reference_shape: tuple[int, ...] | None = None
        _reference_field_name: str
        for fmap in fmaps:
            if fmap is None:
                continue  # pragma: no cover
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

                if require_sorted:
                    _a = data[0]
                    for i, _b in enumerate(data[1:], start=1):
                        if _a > _b:
                            raise ValueError(
                                f"Field {name!r} is not properly sorted by ascending order. "
                                f"Got {_a} (index {i-1}) > {_b} (index {i})"
                            )
                        _a = _b

                if not required_attrs:
                    continue  # pragma: no cover
                for attr, expected in required_attrs.items():
                    if (actual := getattr(data, attr)) != expected:
                        raise ValueError(
                            f"Field {name!r} has incorrect {attr} {actual} "
                            f"(expected {expected})"
                        )

    def _validate_geometry(self) -> None:
        known_axes: dict[Geometry, tuple[Name, Name, Name]] = {
            Geometry.CARTESIAN: ("x", "y", "z"),
            Geometry.POLAR: ("radius", "z", "azimuth"),
            Geometry.CYLINDRICAL: ("radius", "azimuth", "z"),
            Geometry.SPHERICAL: ("radius", "colatitude", "azimuth"),
        }
        if self.geometry not in known_axes:
            # TODO: when Python 3.10 is required, refactor as a match/case block
            # and check that the default case is unreacheable at type check time
            raise ValueError(
                f"Unknown geometry {self.geometry.name.lower()!r}"
            )  # pragma: no cover

        axes = known_axes[self.geometry]
        for expected, actual in zip(axes, self.axes):
            if actual != expected:
                raise ValueError(
                    f"Got invalid axis {actual!r} with geometry {self.geometry.name.lower()!r}"
                )


class CoordinateValidatorMixin(ValidatorMixin, CoordinateData, ABC):
    def _validate_coordinates(self) -> None:
        known_limits: dict[Name, tuple[float | None, float | None]] = {
            "radius": (0, None),
            "azimuth": (0, 2 * np.pi),
            "colatitude": (0, np.pi),
        }
        for axis, coord in self.coordinates.items():
            lims = known_limits.get(axis)
            if lims is None:
                continue
            xmin, xmax = lims
            if xmin is not None and (cmin := np.min(coord)) < xmin:
                raise ValueError(
                    f"Invalid coordinate data for axis {axis!r} {cmin} "
                    f"(minimal allowed value is {xmin})"
                )
            if xmax is not None and (cmax := np.max(coord)) > xmax:
                raise ValueError(
                    f"Invalid coordinate data for axis {axis!r} {cmax} "
                    f"(maximal allowed value is {xmax})"
                )


class Grid(CoordinateValidatorMixin):
    def __init__(
        self,
        geometry: Geometry,
        cell_edges: FieldMap,
        fields: FieldMap | None,
    ) -> None:
        self.geometry = geometry
        self.coordinates = cell_edges
        self.fields = fields

        self.axes = tuple(self.coordinates.keys())
        super().__init__()

    def _validate(self) -> None:
        self._validate_geometry()
        self._validate_coordinates()
        self._validate_fieldmaps(self.cell_edges, ndim=1, require_sorted=True)
        self._validate_fieldmaps(
            self.fields,
            size=self.size,
            ndim=self.ndim,
            shape=self.shape,
        )

    @property
    def cell_edges(self) -> FieldMap:
        return self.coordinates

    @cached_property
    def cell_centers(self) -> FieldMap:
        return {ax: 0.5 * (arr[1:] + arr[:-1]) for ax, arr in self.coordinates.items()}

    @cached_property
    def cell_widths(self) -> FieldMap:
        return {ax: np.diff(arr) for ax, arr in self.coordinates.items()}

    @cached_property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(_) - 1 for _ in self.cell_edges.values())

    @property
    def size(self) -> int:
        return reduce(int.__mul__, self.shape)

    @property
    def ndim(self) -> int:
        return len(self.axes)


class ParticleSet(CoordinateValidatorMixin):
    def __init__(
        self,
        geometry: Geometry,
        coordinates: FieldMap,
        fields: FieldMap | None,
    ) -> None:
        self.geometry = geometry
        self.coordinates = coordinates
        self.fields = fields

        self.axes = tuple(self.coordinates.keys())
        super().__init__()

    def _validate(self) -> None:
        self._validate_geometry()
        self._validate_coordinates()
        self._validate_fieldmaps(
            self.coordinates, self.fields, require_shape_equality=True, ndim=1
        )

    @property
    def count(self) -> int:
        return len(next(iter(self.coordinates.values())))

    @property
    def ndim(self) -> int:
        return len(self.axes)


class Dataset(ValidatorMixin):
    def __init__(
        self,
        *,
        geometry: Geometry = Geometry.CARTESIAN,
        grid: Grid | None = None,
        particles: ParticleSet | None = None,
    ) -> None:
        self.geometry = geometry
        self.grid = grid
        self.particles = particles

        if self.grid is not None:
            self.axes = self.grid.axes
        elif self.particles is not None:
            self.axes = self.particles.axes
        else:
            raise TypeError(
                "Cannot instantiate empty dataset. "
                "Grid and/or particle data must be provided"
            )
        super().__init__()

    def _validate(self) -> None:
        if self.grid is None or self.particles is None:
            return
        for ax, edges in self.grid.cell_edges.items():
            domain_left = edges[0]
            domain_right = edges[-1]
            for x in self.particles.coordinates[ax]:
                if x < domain_left:
                    raise ValueError(f"Got particle at {ax}={x} < {domain_left=}")
                if x > domain_right:
                    raise ValueError(f"Got particle at {ax}={x} > {domain_right=}")

    def _setup_host_cell_index(self) -> None:
        if hasattr(self, "_hci") or self.particles is None or self.grid is None:
            # this line is hard to cover by testing just public api
            # because it's a pure performance optimization with no other observable effect
            return  # pragma: no cover
        from .clib._indexing import _index_particles  # type: ignore [import]

        self._hci = np.empty((self.particles.count, self.grid.ndim), dtype="uint16")

        edges = iter(self.grid.cell_edges.values())

        cell_edges_x1 = next(edges)
        DTYPE = cell_edges_x1.dtype
        cell_edges_x2 = cell_edges_x3 = np.empty(0, DTYPE)
        if self.grid.ndim >= 2:
            cell_edges_x2 = next(edges)
        if self.grid.ndim == 3:
            cell_edges_x3 = next(edges)

        particle_coords = np.empty((self.particles.count, self.grid.ndim), dtype=DTYPE)
        np.stack(
            [e for e in self.particles.coordinates.values()],
            axis=1,
            out=particle_coords,
        )

        _index_particles(
            cell_edges_x1=cell_edges_x1,
            cell_edges_x2=cell_edges_x2,
            cell_edges_x3=cell_edges_x3,
            particle_coords=particle_coords,
            out=self._hci,
        )

    def deposit(self, particle_field_key: Name, /, *, method: Name) -> np.ndarray:
        from .clib._deposition_methods import _deposit_pic_1D  # type: ignore [import]
        from .clib._deposition_methods import _deposit_pic_2D  # type: ignore [import]
        from .clib._deposition_methods import _deposit_pic_3D  # type: ignore [import]

        if not hasattr(self, "_cache"):
            self._cache: dict[tuple[Name, Name], np.ndarray] = {}
        if (particle_field_key, method) in self._cache:
            return self._cache[particle_field_key, method]

        # public interface
        if method in _deposition_method_names:
            key = _deposition_method_names[method]
        else:
            raise ValueError(
                f"Unknown deposition method {method!r}, "
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
        known_methods: dict[DepositionMethod, list[Any]] = {
            DepositionMethod.PARTICLE_IN_CELL: [
                _deposit_pic_1D,
                _deposit_pic_2D,
                _deposit_pic_3D,
            ]
        }
        if key not in known_methods:
            raise NotImplementedError(f"method {method} is not implemented yet")

        field = np.array(self.particles.fields[particle_field_key])
        ret_array = np.zeros(self.grid.shape, dtype=field.dtype)
        self._setup_host_cell_index()

        func = known_methods[key][self.grid.ndim - 1]
        func(field, self._hci, ret_array)
        self._cache[particle_field_key, method] = ret_array
        return ret_array
