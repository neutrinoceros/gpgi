r"""Define the core data structures of the library: Grid, ParticleSet, and Dataset."""

from __future__ import annotations

import enum
import math
import sys
import warnings
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from functools import cached_property, partial, reduce
from itertools import chain
from textwrap import indent
from threading import Lock
from time import monotonic_ns
from typing import TYPE_CHECKING, Literal, assert_never, cast

import numpy as np

from gpgi._boundaries import BoundaryRegistry
from gpgi._lib import (
    _deposit_cic_1D,
    _deposit_cic_2D,
    _deposit_cic_3D,
    _deposit_ngp_1D,
    _deposit_ngp_2D,
    _deposit_ngp_3D,
    _deposit_tsc_1D,
    _deposit_tsc_2D,
    _deposit_tsc_3D,
    _index_particles,
)
from gpgi._typing import FieldMap, Name
from gpgi.typing import DepositionMethodT, DepositionMethodWithMetadataT

if sys.version_info >= (3, 13):
    LockType = Lock
else:
    from _thread import LockType

if TYPE_CHECKING:
    from typing import Any, Self

    from gpgi._typing import FieldMap, HCIArray, Name, RealArray

BoundarySpec = tuple[tuple[str, str, str], ...]


class Geometry(enum.StrEnum):
    CARTESIAN = enum.auto()
    POLAR = enum.auto()
    CYLINDRICAL = enum.auto()
    SPHERICAL = enum.auto()
    EQUATORIAL = enum.auto()


class DepositionMethod(enum.Enum):
    NEAREST_GRID_POINT = enum.auto()
    CLOUD_IN_CELL = enum.auto()
    TRIANGULAR_SHAPED_CLOUD = enum.auto()


_deposition_method_names: dict[str, DepositionMethod] = {
    **{m.name.lower(): m for m in DepositionMethod},
    **{"".join([w[0] for w in m.name.split("_")]).lower(): m for m in DepositionMethod},
}


_BUILTIN_METHODS: dict[DepositionMethod, list[DepositionMethodT]] = {
    DepositionMethod.NEAREST_GRID_POINT: [
        _deposit_ngp_1D,
        _deposit_ngp_2D,
        _deposit_ngp_3D,
    ],
    DepositionMethod.CLOUD_IN_CELL: [
        _deposit_cic_1D,
        _deposit_cic_2D,
        _deposit_cic_3D,
    ],
    DepositionMethod.TRIANGULAR_SHAPED_CLOUD: [
        _deposit_tsc_1D,
        _deposit_tsc_2D,
        _deposit_tsc_3D,
    ],
}


class GeometricData(ABC):
    geometry: Geometry
    axes: tuple[Name, ...]


class CoordinateData(ABC):
    geometry: Geometry
    axes: tuple[Name, ...]
    coordinates: FieldMap
    fields: FieldMap


class ValidatorMixin(GeometricData, ABC):
    def __init__(self) -> None:
        self._validate()

    @abstractmethod
    def _validate(self) -> None: ...

    def _validate_FieldMaps(
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
        match self.geometry:
            case Geometry.CARTESIAN:
                axes3D = ("x", "y", "z")
            case Geometry.POLAR:
                axes3D = ("radius", "azimuth", "z")
            case Geometry.CYLINDRICAL:
                axes3D = ("radius", "z", "azimuth")
            case Geometry.SPHERICAL:
                axes3D = ("radius", "colatitude", "azimuth")
            case Geometry.EQUATORIAL:
                axes3D = ("radius", "azimuth", "latitude")
            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

        for i, (expected, actual) in enumerate(zip(axes3D, self.axes, strict=False)):
            if actual != expected:
                raise ValueError(
                    f"Got invalid axis name {actual!r} on position {i}, "
                    f"with geometry {self.geometry.name.lower()!r}\n"
                    f"Expected axes ordered as {axes3D[: len(self.axes)]}"
                )


_AXES_LIMITS: dict[Name, tuple[float, float]] = {
    "x": (-float("inf"), float("inf")),
    "y": (-float("inf"), float("inf")),
    "z": (-float("inf"), float("inf")),
    "radius": (0, float("inf")),
    "azimuth": (0, 2 * np.pi),
    "colatitude": (0, np.pi),
    "latitude": (-np.pi / 2, np.pi / 2),
}


class _CoordinateValidatorMixin(ValidatorMixin, CoordinateData, ABC):
    def __init__(self) -> None:
        super().__init__()
        dts = {
            name: arr.dtype
            for name, arr in chain(
                self.coordinates.items(),
                self.fields.items(),
            )
        }
        unique_dts = sorted(set(dts.values()))
        if len(unique_dts) > 1:
            raise TypeError(f"Grid received mixed data types ({unique_dts}):\n{dts}")
        else:
            self.dtype = unique_dts.pop()

    def _validate_coordinates(self) -> None:
        for axis in self.axes:
            coord = self.coordinates[axis]
            if len(coord) == 0:
                continue
            coord_dtype = self._get_safe_datatype(coord)
            dt = coord_dtype.type
            xmin, xmax = (dt(_) for _ in _AXES_LIMITS[axis])
            if (cmin := dt(np.min(coord))) < xmin or not math.isfinite(cmin):
                if math.isfinite(xmin):
                    hint = f"minimal value allowed is {xmin}"
                else:
                    assert xmin == -float("inf")
                    hint = "value must be finite"
                raise ValueError(
                    f"Invalid coordinate data for axis {axis!r} {cmin} ({hint})"
                )
            if (cmax := dt(np.max(coord))) > xmax or not math.isfinite(cmax):
                if math.isfinite(xmax):
                    hint = f"maximal value allowed is {xmax}"
                else:
                    assert xmax == float("inf")
                    hint = "value must be finite"
                raise ValueError(
                    f"Invalid coordinate data for axis {axis!r} {cmax} ({hint})"
                )

            self.coordinates[axis] = coord.astype(coord_dtype, copy=False)

    def _get_safe_datatype(self, reference: np.ndarray | None = None) -> np.dtype:
        if reference is None:
            reference = self.coordinates[self.axes[0]]
        dt = reference.dtype
        if dt.kind != "f":
            raise ValueError(f"Invalid data type {dt} (expected a float dtype)")
        # return type should already be correct but
        # this has the benefit of convincing mypy
        return np.dtype(dt)


class Grid(_CoordinateValidatorMixin):
    def __init__(
        self,
        *,
        geometry: Geometry,
        cell_edges: FieldMap,
        fields: FieldMap | None = None,
    ) -> None:
        r"""
        Define a Grid from cell left-edges and data fields.

        Parameters
        ----------
        geometry (keyword-only): gpgi.Geometry

        cell_edges (keyword-only): gpgi.typing.FieldMap
            Left cell edges in each direction as 1D arrays, including the right edge
            of the rightmost cell.

        fields (keyword-only, optional): gpgi.typing.FieldMap
        """
        self.geometry = geometry
        self.coordinates = cell_edges

        if fields is None:
            fields = {}
        self.fields: FieldMap = fields

        self.axes = tuple(self.coordinates.keys())
        super().__init__()

        self._dx = np.full((3,), -1, dtype=self.coordinates[self.axes[0]].dtype)
        for i, ax in enumerate(self.axes):
            if self.size == 1 or np.diff(self.coordinates[ax]).std() < 1e-16:
                # got a constant step in this direction, store it
                self._dx[i] = self.coordinates[ax][1] - self.coordinates[ax][0]

    def __repr__(self) -> str:
        """Implement repr(Grid(...))."""
        return (
            f"{self.__class__.__name__}(\n"
            f"    geometry={str(self.geometry)!r},\n"
            f"    cell_edges={self.coordinates},\n"
            f"    fields={self.fields},\n"
            ")"
        )

    def _validate(self) -> None:
        self._validate_geometry()
        self._validate_coordinates()
        self._validate_FieldMaps(self.cell_edges, ndim=1, require_sorted=True)
        self._validate_FieldMaps(
            self.fields,
            size=self.size,
            ndim=self.ndim,
            shape=self.shape,
        )

    @property
    def cell_edges(self) -> FieldMap:
        r"""An alias for self.coordinates."""
        return self.coordinates

    @cached_property
    def cell_centers(self) -> FieldMap:
        r"""The positions of cell centers in each direction."""
        return {ax: 0.5 * (arr[1:] + arr[:-1]) for ax, arr in self.coordinates.items()}

    @cached_property
    def cell_widths(self) -> FieldMap:
        r"""The width of cells, expressed as the difference between consecutive left edges."""
        return {ax: np.diff(arr) for ax, arr in self.coordinates.items()}

    @cached_property
    def shape(self) -> tuple[int, ...]:
        r"""The shape of the grid, as a tuple (nx1, (nx2, (nx3)))."""
        return tuple(len(_) - 1 for _ in self.cell_edges.values())

    @cached_property
    def _padded_shape(self) -> tuple[int, ...]:
        return tuple(len(_) + 1 for _ in self.cell_edges.values())

    @property
    def size(self) -> int:
        r"""The total number of cells in the grid."""
        return reduce(int.__mul__, self.shape)

    @property
    def ndim(self) -> int:
        r"""The number of spatial dimensions that coordinates are defined in."""
        return len(self.axes)

    @property
    def cell_volumes(self) -> RealArray:
        r"""
        The generalized ND-volume of grid cells.

        3D: (nx, ny, nz) array representing volumes
        2D: (nx, ny) array representing surface
        1D: (nx,) array representing widths (redundant with cell_widths).
        """
        widths = list(self.cell_widths.values())
        if self.geometry is Geometry.CARTESIAN:
            raw = np.prod(np.meshgrid(*widths), axis=0)
            return cast("RealArray", np.swapaxes(raw, 0, 1))
        else:
            raise NotImplementedError(
                f"cell_volumes property is not implemented for {self.geometry} geometry"
            )


class ParticleSet(_CoordinateValidatorMixin):
    def __init__(
        self,
        *,
        geometry: Geometry,
        coordinates: FieldMap,
        fields: FieldMap | None = None,
    ) -> None:
        r"""
        Define a ParticleSet from point positions and data fields.

        Parameters
        ----------
        geometry (keyword-only): gpgi.Geometry

        coordinates (keyword-only): gpgi.typing.FieldMap
            Particle positions in each direction as 1D arrays.

        fields (keyword-only, optional): gpgi.typing.FieldMap
        """
        self.geometry = geometry
        self.coordinates = coordinates

        if fields is None:
            fields = {}
        self.fields: FieldMap = fields

        self.axes = tuple(self.coordinates.keys())
        super().__init__()

    def __repr__(self) -> str:
        """Implement repr(ParticleSet(...))."""
        return (
            f"{self.__class__.__name__}(\n"
            f"    geometry={str(self.geometry)!r},\n"
            f"    coordinates={self.coordinates},\n"
            f"    fields={self.fields},\n"
            ")"
        )

    def _validate(self) -> None:
        self._validate_geometry()
        self._validate_coordinates()
        self._validate_FieldMaps(
            self.coordinates, self.fields, require_shape_equality=True, ndim=1
        )

    @property
    def count(self) -> int:
        r"""The total number of particles in the set."""
        return len(next(iter(self.coordinates.values())))

    @property
    def ndim(self) -> int:
        r"""The number of spatial dimensions that coordinates are defined in."""
        return len(self.axes)


class Dataset(ValidatorMixin):
    def __init__(
        self,
        *,
        geometry: Geometry = Geometry.CARTESIAN,
        grid: Grid,
        particles: ParticleSet | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        r"""
        Compose a Dataset from a Grid, a ParticleSet, or both.

        Parameters
        ----------
        geometry (keyword-only): gpgi.Geometry
            An enum member that represents the geometry.

        grid (keyword-only): gpgi.Grid

        particles(keyword-only, optional): gpgi.ParticleSet

        metadata (keyword-only, optional): dict[str, Any]
            A dictionary representing arbitrary additional data, that will be attached
            to the returned Dataset as an attribute (namely, ds.metadata). This special
            attribute is accessible from boundary condition methods as the argument of
            the same name.

            .. versionadded: 0.4.0
        """
        self.geometry = geometry

        if particles is None:
            dt = grid._get_safe_datatype()
            particles = ParticleSet(
                geometry=grid.geometry,
                coordinates={ax: np.array([], dtype=dt) for ax in grid.axes},
            )

        self.grid: Grid = grid
        self.particles: ParticleSet = particles

        self.boundary_recipes = BoundaryRegistry()
        self.axes = self.grid.axes
        self.metadata = deepcopy(metadata) if metadata is not None else {}

        self._hci: HCIArray | None = None
        self._hci_lock = Lock()
        self._deposit_lock = Lock()

        super().__init__()

    def __repr__(self) -> Name:
        """Implement repr(Dataset(...))."""
        return (
            f"{self.__class__.__name__}(\n"
            f"    geometry={str(self.geometry)!r},\n"
            f"    grid={indent(str(self.grid), ' '*4).lstrip()},\n"
            f"    particles={indent(str(self.particles), ' '*4).lstrip()},\n"
            f"    metadata={self.metadata},\n"
            ")"
        )

    def _validate(self) -> None:
        if self.particles.count == 0:
            return
        for ax, edges in self.grid.cell_edges.items():
            domain_left = float(edges[0])
            domain_right = float(edges[-1])
            for x in self.particles.coordinates[ax]:
                if x < domain_left:
                    raise ValueError(f"Got particle at {ax}={x} < {domain_left=}")
                if x > domain_right:
                    raise ValueError(f"Got particle at {ax}={x} > {domain_right=}")

        if self.particles.dtype != self.grid.dtype:
            raise TypeError(
                f"Got mixed data types:\n"
                f"- from grid data: {self.grid.dtype}\n"
                f"- from particles: {self.particles.dtype}\n"
            )

    def _get_padded_cell_edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        edges = iter(self.grid.cell_edges.values())

        def pad(a: np.ndarray) -> np.ndarray:
            dx = a[1] - a[0]
            return np.concatenate([[a[0] - dx], a, [a[-1] + dx]])

        x1 = next(edges)
        cell_edges_x1 = pad(x1)
        DTYPE = cell_edges_x1.dtype

        cell_edges_x2 = cell_edges_x3 = np.empty(0, DTYPE)
        if self.grid.ndim >= 2:
            cell_edges_x2 = pad(next(edges))
        if self.grid.ndim == 3:
            cell_edges_x3 = pad(next(edges))

        return cell_edges_x1, cell_edges_x2, cell_edges_x3

    def _get_3D_particle_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        particle_coords = iter(self.particles.coordinates.values())
        particles_x1 = next(particle_coords)
        DTYPE = particles_x1.dtype

        particles_x2 = particles_x3 = np.empty(0, DTYPE)
        if self.grid.ndim >= 2:
            particles_x2 = next(particle_coords)
        if self.grid.ndim == 3:
            particles_x3 = next(particle_coords)

        return particles_x1, particles_x2, particles_x3

    def _compute_host_cell_index(self, verbose: bool = False) -> HCIArray:
        hci = np.empty((self.particles.count, self.grid.ndim), dtype="uint16")

        tstart = monotonic_ns()
        _index_particles(
            *self._get_padded_cell_edges(),
            *self._get_3D_particle_coordinates(),
            dx=self.grid._dx,
            out=hci,
        )
        for idim in range(hci.shape[1]):
            # There are at leas two edge cases where we need clipping to correct raw indices:
            # - particles that live exactly on the domain right edge will be indexed out of bound
            # - single precision positions can lead to overshoots by simple effect of floating point arithmetics
            #   (double precision isn't safe either, it's just safer (by a lot))
            hci[:, idim].clip(1, self.grid.shape[idim], out=hci[:, idim])

        tstop = monotonic_ns()
        if verbose:
            print(
                f"Indexed {self.particles.count:.4g} particles in {(tstop-tstart)/1e9:.2f} s"
            )
        return hci

    def _setup_host_cell_index(self, verbose: bool = False) -> HCIArray:
        """Pre-compute internal host cell index array, used for depostition.

        This method is thread-safe.
        The result is returned for testing/typechecking convenience.
        """
        # the first thread to acquire the lock does the computation, all other
        # threads can skip it and reuse the result
        with self._hci_lock:
            if self._hci is None:
                self._hci = self._compute_host_cell_index(verbose)
        return self._hci

    @property
    def host_cell_index(self) -> HCIArray:
        r"""
        The ND index of the host cell for each particle.

        It has shape (particles.count, grid.ndim).
        Indices are 0-based and ghost layers are included.
        """
        if self._hci is None:
            # rebinding for typechecking convenience only
            self._hci = self._setup_host_cell_index()
        return self._hci

    @property
    def _default_sort_axes(self) -> tuple[int, ...]:
        return tuple(range(self.grid.ndim - 1, -1, -1))

    def _validate_sort_axes(self, axes: tuple[int, ...]) -> None:
        if len(axes) != self.grid.ndim:
            raise ValueError(f"Expected exactly {self.grid.ndim} axes, got {len(axes)}")
        if any(not isinstance(axis, int) for axis in axes):
            raise ValueError(
                f"Expected all axes to be integers, got {axes!r} "
                f"with types ({', '.join([type(axis).__name__ for axis in axes])})"
            )
        if any(axis > self.grid.ndim - 1 for axis in axes):
            raise ValueError(f"Expected all axes to be <{self.grid.ndim}, got {axes!r}")

    def _get_sort_key(
        self, axes: tuple[int, ...]
    ) -> np.ndarray[Any, np.dtype[np.uint16]]:
        self._validate_sort_axes(axes)

        hci = self.host_cell_index
        if self.grid.ndim == 1:
            ind = np.lexsort((hci[:, axes[0]],))
        elif self.grid.ndim == 2:
            ind = np.lexsort((hci[:, axes[0]], hci[:, axes[1]]))
        else:
            ind = np.lexsort((hci[:, axes[0]], hci[:, axes[1]], hci[:, axes[2]]))
        return np.array(ind, dtype="uint16")

    def is_sorted(self, *, axes: tuple[int, ...] | None = None) -> bool:
        r"""
        Return True if and only if particles are already sorted.

        .. versionadded: 0.14.0

        Parameters
        ----------
        axes (keyword-only, optional): tuple[int, ...]
            specify in which order axes should be used for sorting.
        """
        sort_key = self._get_sort_key(axes or self._default_sort_axes)
        hci = self.host_cell_index
        return bool(np.all(hci == hci[sort_key]))

    def sorted(self, *, axes: tuple[int, ...] | None = None) -> Self:
        r"""
        Return a copy of this dataset with particles sorted by host cell index.

        .. versionadded: 0.14.0

        Parameters
        ----------
        axes (keyword-only, optional): tuple[int, ...]
            specify in which order axes should be used for sorting.
        """
        sort_key = self._get_sort_key(axes or self._default_sort_axes)

        return type(self)(
            geometry=self.geometry,
            grid=self.grid,
            particles=ParticleSet(
                geometry=self.geometry,
                coordinates={
                    name: arr[sort_key]
                    for name, arr in deepcopy(self.particles.coordinates).items()
                },
                fields={
                    name: arr[sort_key]
                    for name, arr in deepcopy(self.particles.fields).items()
                },
            ),
        )

    def deposit(
        self,
        particle_field_key: Name,
        /,
        *,
        method: Literal[
            "ngp",
            "nearest_grid_point",
            "cic",
            "cloud_in_cell",
            "tsc",
            "triangular_shaped_cloud",
        ]
        | DepositionMethodT
        | DepositionMethodWithMetadataT,
        boundaries: dict[Name, tuple[Name, Name]] | None = None,
        verbose: bool = False,
        return_ghost_padded_array: bool = False,
        weight_field: Name | None = None,
        weight_field_boundaries: dict[Name, tuple[Name, Name]] | None = None,
        lock: Literal["per-instance"] | None | LockType = "per-instance",
    ) -> np.ndarray:
        r"""
        Perform particle deposition and return the result as a grid field.

        Parameters
        ----------
        particle_field_key (positional only): str
            label of the particle field to deposit

        method (keyword only): 'ngp', 'cic' or 'tsc', or function
            full names ('nearest_grid_point', 'cloud_in_cell', and
            'triangular_shaped_cloud') are also valid

            .. versionchanged:: 0.12.0
                Added support for user-defined functions.

        verbose (keyword only, optional): bool (default False)
            if True, print execution time for hot loops (indexing and deposition)

        return_ghost_padded_array (keyword only, optional): bool (default False)
            if True, return the complete deposition array, including one extra
            cell layer per direction and per side. This option is meant as a
            debugging tool for methods that leak some particle data outside the
            active domain (cic and tsc).

        weight_field (keyword only, optional): str
            label of another field to use as weights. Let u be the field to
            deposit and w be the weight field. Let u' and w' be their equivalent
            on-grid descriptions. u' is obtained as

            w'(x) = Σ w(i) c(i,x)
            u'(x) = (1/w'(x)) Σ u(i) w(i) c(i,x)

            where x is the spatial position, i is a particle index, and w(i,x)
            are geometric coefficients associated with the deposition method.

            .. versionadded: 0.7.0

        boundaries and weight_field_boundaries (keyword only, optional): dict
            Maps from axis names (str) to boundary recipe keys (str, str)
            representing left/right boundaries. By default all axes will use
            'open' boundaries on both sides. Specifying boundaries for all axes
            is not mandated, but note that recipes are applied in the order of
            specified axes (any unspecified axes will be treated last).

            weight_field_boundaries is required if weight field is used in
            combinations with boundaries.

            Boundary recipes are applied the weight field (if any) first.

            .. versionadded: 0.5.0

        lock (keyword only, optional): 'per-instance' (default), None, or threading.Lock
            Fine tune performance for multi-threaded applications: define a
            locking strategy around the deposition hotloop.
            - 'per-instance': allow multiple Dataset instances to run deposition
                concurrently, but forbid concurrent accesses to any specific
                instance
            - None: no locking is applied. Within some restricted conditions
                (e.g. depositing a couple fields concurrently in a sorted dataset),
                this may improve walltime performance, but it is also expected to
                degrade it in a more general case as it encourages cache-misses
            - an arbitrary threading.Lock instance may be supplied to implement
                a custom strategy

            .. versionadded:: 2.0.0
        """
        if callable(method):
            from inspect import signature

            sig = signature(method)
            func: DepositionMethodT
            if "metadata" in sig.parameters:
                method = cast(DepositionMethodWithMetadataT, method)
                func = partial(method, metadata=self.metadata)
            else:
                method = cast(DepositionMethodT, method)
                func = method
        else:
            if method not in _deposition_method_names:
                raise ValueError(
                    f"Unknown deposition method {method!r}, "
                    f"expected any of {tuple(_deposition_method_names.keys())}"
                )

            if (mkey := _deposition_method_names[method]) not in _BUILTIN_METHODS:
                raise NotImplementedError(f"method {method} is not implemented yet")

            func = _BUILTIN_METHODS[mkey][self.grid.ndim - 1]

        if self.particles.count == 0:
            raise TypeError("Cannot deposit particle fields on a particle-less dataset")
        if not self.particles.fields:
            raise TypeError("There are no particle fields")
        if particle_field_key not in self.particles.fields:
            raise ValueError(f"Unknown particle field {particle_field_key!r}")

        if boundaries is None:
            boundaries = {}
        if weight_field_boundaries is None:
            if boundaries and weight_field is not None:
                raise TypeError(
                    "weight_field_boundaries keyword argument is "
                    "required with weight_field and boundaries"
                )
            weight_field_boundaries = {}
        elif weight_field is None:
            warnings.warn(
                "weight_field_boundaries will not be used "
                "as no weight_field was specified",
                stacklevel=2,
            )

        self._sanitize_boundaries(boundaries)
        self._sanitize_boundaries(weight_field_boundaries)

        lock_ctx: AbstractContextManager
        match lock:
            case "per-instance":
                lock_ctx = self._deposit_lock
            case None:
                lock_ctx = nullcontext()
            case LockType():
                lock_ctx = lock
            case _:
                raise ValueError(
                    f"Received {lock=!r}. Expected either 'per-instance', "
                    "None, or an instance of threading.Lock"
                )

        field = self.particles.fields[particle_field_key]
        padded_ret_array = np.zeros(self.grid._padded_shape, dtype=field.dtype)
        if weight_field is not None:
            wfield = np.array(self.particles.fields[weight_field])
            wfield_dep = np.zeros(self.grid._padded_shape, dtype=field.dtype)
        else:
            wfield = np.array((), dtype=field.dtype)
            wfield_dep = np.array((1,), dtype=field.dtype)

        # rebinding for typechecking convenience only
        self._hci = self._setup_host_cell_index(verbose)

        tstart = monotonic_ns()
        with lock_ctx:
            if weight_field is not None:
                func(
                    *self._get_padded_cell_edges(),
                    *self._get_3D_particle_coordinates(),
                    wfield,
                    np.array((), dtype=field.dtype),
                    self._hci,
                    wfield_dep,
                )

            func(
                *self._get_padded_cell_edges(),
                *self._get_3D_particle_coordinates(),
                field,
                wfield,
                self._hci,
                padded_ret_array,
            )

        tstop = monotonic_ns()
        if verbose:
            print(
                f"Deposited {self.particles.count:.4g} particles in {(tstop-tstart)/1e9:.2f} s"
            )

        if weight_field is not None:
            self._apply_boundary_conditions(
                array=wfield_dep,
                boundaries=weight_field_boundaries,
                weight_array=None,
            )

            self._apply_boundary_conditions(
                array=padded_ret_array,
                boundaries=boundaries,
                weight_array=wfield_dep,
            )
        else:
            self._apply_boundary_conditions(
                array=padded_ret_array,
                boundaries=boundaries,
                weight_array=None,
            )

        with np.errstate(invalid="ignore"):
            padded_ret_array /= wfield_dep

        if return_ghost_padded_array:
            return padded_ret_array
        elif self.grid.ndim == 1:
            return padded_ret_array[1:-1]
        elif self.grid.ndim == 2:
            return padded_ret_array[1:-1, 1:-1]
        elif self.grid.ndim == 3:
            return padded_ret_array[1:-1, 1:-1, 1:-1]
        else:  # pragma: no cover
            raise RuntimeError("Caching error. Please report this.")

    def _sanitize_boundaries(self, boundaries: dict[Name, tuple[Name, Name]]) -> None:
        for ax in self.grid.axes:
            boundaries.setdefault(ax, ("open", "open"))
        for axk in boundaries:
            if axk not in self.grid.axes:
                raise ValueError(
                    f"Got invalid ax key {axk!r}, expected any of {self.grid.axes!r}"
                )
        for bound in boundaries.values():
            if (
                (not isinstance(bound, tuple))
                or len(bound) != 2
                or not all(isinstance(_, str) for _ in bound)
            ):
                raise TypeError(f"Expected a 2-tuple of strings, got {bound!r}")

            for b in bound:
                if b not in self.boundary_recipes:
                    raise ValueError(f"Unknown boundary type {b!r}")

    def _apply_boundary_conditions(
        self,
        array: RealArray,
        boundaries: dict[Name, tuple[Name, Name]],
        weight_array: RealArray | None,
    ) -> None:
        axes = list(self.grid.axes)
        for ax, bv in boundaries.items():
            # in some applications, it is *crucial* that boundaries be applied
            # in a specific order, so we take advantage of the fact that dictionaries
            # are ordered in modern Python and loop over user-specified constraints
            # as a way to *expose* ordering.
            iax = axes.index(ax)
            bcs = tuple(self.boundary_recipes[key] for key in bv)
            for side, bc in zip(("left", "right"), bcs, strict=True):
                side = cast(Literal["left", "right"], side)
                active_index: int = 1 if side == "left" else -2
                same_side_active_layer_idx = [slice(None)] * self.grid.ndim
                same_side_active_layer_idx[iax] = active_index  # type:ignore [call-overload]

                same_side_ghost_layer_idx = [slice(None)] * self.grid.ndim
                # f(-2)=-1, f(1)=0
                same_side_ghost_layer_idx[iax] = -((active_index + 1) % 2)  # type:ignore [call-overload]

                opposite_side_active_layer_idx = [slice(None)] * self.grid.ndim
                # f(-2)=1, f(1)=-2
                opposite_side_active_layer_idx[iax] = (  # type:ignore [call-overload]
                    1 if active_index == -2 else -2
                )

                opposite_side_ghost_layer_idx = [slice(None)] * self.grid.ndim
                # f(-2)=0, f(1)=-1
                opposite_side_ghost_layer_idx[iax] = -(active_index % 2)  # type:ignore [call-overload]

                if weight_array is None:
                    ONE = np.array((1,), dtype=array.dtype)
                    array[tuple(same_side_active_layer_idx)] = bc(
                        array[tuple(same_side_active_layer_idx)],
                        array[tuple(same_side_ghost_layer_idx)],
                        array[tuple(opposite_side_active_layer_idx)],
                        array[tuple(opposite_side_ghost_layer_idx)],
                        ONE,
                        ONE,
                        ONE,
                        ONE,
                        side,
                        self.metadata,
                    )
                else:
                    array[tuple(same_side_active_layer_idx)] = bc(
                        array[tuple(same_side_active_layer_idx)],
                        array[tuple(same_side_ghost_layer_idx)],
                        array[tuple(opposite_side_active_layer_idx)],
                        array[tuple(opposite_side_ghost_layer_idx)],
                        weight_array[tuple(same_side_active_layer_idx)],
                        weight_array[tuple(same_side_ghost_layer_idx)],
                        weight_array[tuple(opposite_side_active_layer_idx)],
                        weight_array[tuple(opposite_side_ghost_layer_idx)],
                        side,
                        self.metadata,
                    )
