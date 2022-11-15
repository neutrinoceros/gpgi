from __future__ import annotations

import enum
import sys
import warnings
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from functools import cached_property
from functools import reduce
from itertools import chain
from time import monotonic_ns
from typing import Any
from typing import cast
from typing import Dict
from typing import Literal
from typing import Protocol
from typing import Tuple
from typing import TYPE_CHECKING

import numpy as np

from ._boundaries import BoundaryRegistry

if TYPE_CHECKING:
    from ._typing import HCIArray
    from ._typing import RealArray

if sys.version_info >= (3, 9):
    from collections.abc import Callable
else:
    from typing import Callable


BoundarySpec = Tuple[Tuple[str, str, str], ...]


class Geometry(enum.Enum):
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


Name = str
FieldMap = Dict[Name, np.ndarray]
DepositionMethodT = Callable[
    [
        "RealArray",
        "RealArray",
        "RealArray",
        "RealArray",
        "RealArray",
        "RealArray",
        "RealArray",
        "RealArray",
        "HCIArray",
        "RealArray",
    ],
    None,
]


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
            Geometry.EQUATORIAL: ("radius", "azimuth", "latitude"),
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
        fields: FieldMap,
    ) -> None:
        self.geometry = geometry
        self.coordinates = cell_edges
        self.fields = fields

        self.axes = tuple(self.coordinates.keys())
        super().__init__()

        self._dx = np.full((3,), -1, dtype=self.coordinates[self.axes[0]].dtype)
        for i, ax in enumerate(self.axes):
            if np.diff(self.coordinates[ax]).std() < 1e-16:
                # got a constant step in this direction, store it
                self._dx[i] = self.coordinates[ax][1] - self.coordinates[ax][0]

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

    @cached_property
    def _padded_shape(self) -> tuple[int, ...]:
        return tuple(len(_) + 1 for _ in self.cell_edges.values())

    @property
    def size(self) -> int:
        return reduce(int.__mul__, self.shape)

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def cell_volumes(self) -> RealArray:
        """
        3D: (nx, ny, nz) array representing volumes
        2D: (nx, ny) array representing surface
        1D: (nx,) array representing widths (redundant with cell_widths)
        """
        widths = list(self.cell_widths.values())
        if self.geometry is Geometry.CARTESIAN:
            return cast("RealArray", np.prod(np.meshgrid(*widths), axis=0))
        else:
            raise NotImplementedError(
                f"cell_volumes property is not implemented for {self.geometry}"
            )


class ParticleSet(CoordinateValidatorMixin):
    def __init__(
        self,
        geometry: Geometry,
        coordinates: FieldMap,
        fields: FieldMap,
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.geometry = geometry
        self.grid = grid
        self.particles = particles
        self.boundary_recipes = BoundaryRegistry()

        if self.grid is not None:
            self.axes = self.grid.axes
        elif self.particles is not None:
            self.axes = self.particles.axes
        else:
            raise TypeError(
                "Cannot instantiate empty dataset. "
                "Grid and/or particle data must be provided"
            )
        self.metadata = deepcopy(metadata) if metadata is not None else {}

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

        unique_dts = sorted(
            {
                arr.dtype
                for arr in chain(
                    self.grid.coordinates.values(),
                    self.grid.fields.values(),
                    self.particles.coordinates.values(),
                    self.particles.fields.values(),
                )
            }
        )
        if len(unique_dts) > 1:
            raise TypeError(f"Got mixed data types ({unique_dts})")

    def _get_padded_cell_edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.grid is None:
            raise RuntimeError(
                "Something took a wrong turn, please report this."
            )  # pragma: no cover
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
        if self.grid is None or self.particles is None:
            raise RuntimeError(
                "Something took a wrong turn, please report this."
            )  # pragma: no cover

        particle_coords = iter(self.particles.coordinates.values())
        particles_x1 = next(particle_coords)
        DTYPE = particles_x1.dtype

        particles_x2 = particles_x3 = np.empty(0, DTYPE)
        if self.grid.ndim >= 2:
            particles_x2 = next(particle_coords)
        if self.grid.ndim == 3:
            particles_x3 = next(particle_coords)

        return particles_x1, particles_x2, particles_x3

    def _setup_host_cell_index(self, verbose: bool = False) -> None:
        if hasattr(self, "_hci") or self.particles is None or self.grid is None:
            # this line is hard to cover by testing just public api
            # because it's a pure performance optimization with no other observable effect
            return  # pragma: no cover
        from .clib._indexing import _index_particles  # type: ignore [import]

        self._hci = np.empty((self.particles.count, self.grid.ndim), dtype="uint16")

        tstart = monotonic_ns()
        _index_particles(
            *self._get_padded_cell_edges(),
            *self._get_3D_particle_coordinates(),
            dx=self.grid._dx,
            out=self._hci,
        )
        tstop = monotonic_ns()
        if verbose:
            print(
                f"Indexed {self.particles.count:.4g} particles in {(tstop-tstart)/1e9:.2f} s"
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
        ],
        boundaries: dict[Name, tuple[Name, Name]] | None = None,
        verbose: bool = False,
        return_ghost_padded_array: bool = False,
        weight_field: Name | None = None,
        weight_field_boundaries: dict[Name, tuple[Name, Name]] | None = None,
    ) -> np.ndarray:
        r"""
        Perform particle deposition and return the result as a grid field.

        Parameters
        ----------
        particle_field_key (positional only): str
           label of the particle field to deposit

        method (keyword only): 'ngp', 'cic' or 'tsc'
           full names ('nearest_grid_point', 'cloud_in_cell', and
           'triangular_shaped_cloud') are also valid

        verbose (keyword only): bool (default False)
           if True, print execution time for hot loops (indexing and deposition)

        return_ghost_padded_array (keyword only): bool (default False)
           if True, return the complete deposition array, including one extra
           cell layer per direction and per side. This option is meant as a
           debugging tool for methods that leak some particle data outside the
           active domain (cic and tsc).

        weight_field (keyword only): str
           label of another field to use as weights. Let u be the field to
           deposit and w be the weight field. Let u' and w' be their equivalent
           on-grid descriptions. u' is obtained as

           w'(x) = Σ w(i) c(i,x)
           u'(x) = (1/w'(x)) Σ u(i) w(i) c(i,x)

           where x is the spatial position, i is a particle index, and w(i,x)
           are geometric coefficients associated with the deposition method.

        boundaries and weigth_field_boundaries (keyword only): dict
           Maps from axis names (str) to boundary recipe keys (str, str)
           representing left/right boundaries. By default all axes will use
           'open' boundaries on both sides. Specifying boundaries for all axes
           is not mandated, but note that recipes are applied in the order of
           specified axes (any unspecified axes will be treated last).

           weight_field_boundaries is required if weight field is used in
           combinations with boundaries.

           Boundary recipes are applied the weight field (if any) first.
        """
        from .clib._deposition_methods import _deposit_ngp_1D  # type: ignore [import]
        from .clib._deposition_methods import _deposit_ngp_2D  # type: ignore [import]
        from .clib._deposition_methods import _deposit_ngp_3D  # type: ignore [import]
        from .clib._deposition_methods import _deposit_cic_1D  # type: ignore [import]
        from .clib._deposition_methods import _deposit_cic_2D  # type: ignore [import]
        from .clib._deposition_methods import _deposit_cic_3D  # type: ignore [import]
        from .clib._deposition_methods import _deposit_tsc_1D  # type: ignore [import]
        from .clib._deposition_methods import _deposit_tsc_2D  # type: ignore [import]
        from .clib._deposition_methods import _deposit_tsc_3D  # type: ignore [import]

        if method in ("pic", "particle_in_cell"):
            warnings.warn(
                f"{method=!r} is a deprecated alias for method='ngp', "
                "please use 'ngp' (or 'nearest_grid_point') directly",
                DeprecationWarning,
                stacklevel=2,
            )
            method = "ngp"

        if method in _deposition_method_names:
            mkey = _deposition_method_names[method]
        else:
            raise ValueError(
                f"Unknown deposition method {method!r}, "
                f"expected any of {tuple(_deposition_method_names.keys())}"
            )

        if self.grid is None:
            raise TypeError("Cannot deposit particle fields on a grid-less dataset")
        if self.particles is None:
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

        known_methods: dict[DepositionMethod, list[DepositionMethodT]] = {
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
        if mkey not in known_methods:
            raise NotImplementedError(f"method {method} is not implemented yet")

        field = self.particles.fields[particle_field_key]
        padded_ret_array = np.zeros(self.grid._padded_shape, dtype=field.dtype)
        if weight_field is not None:
            wfield = np.array(self.particles.fields[weight_field])
            wfield_dep = np.zeros(self.grid._padded_shape, dtype=field.dtype)
        else:
            wfield = np.ones(0, dtype=field.dtype)
            wfield_dep = np.ones(1, dtype=field.dtype)

        self._setup_host_cell_index(verbose=verbose)

        func = known_methods[mkey][self.grid.ndim - 1]
        tstart = monotonic_ns()
        if weight_field is not None:
            func(
                *self._get_padded_cell_edges(),
                *self._get_3D_particle_coordinates(),
                wfield,
                np.ones(0, dtype=field.dtype),
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
        if self.grid is None:  # pragma: no cover
            raise RuntimeWarning("Something took a wrong turn, please report this")

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
        if self.grid is None:  # pragma: no cover
            raise RuntimeWarning("Something took a wrong turn, please report this")

        axes = list(self.grid.axes)
        for ax, bv in boundaries.items():
            # in some applications, it is *crucial* that boundaries be applied
            # in a specific order, so we take advantage of the fact that dictionaries
            # are ordered in modern Python and loop over user-specified constraints
            # as a way to *expose* ordering.
            iax = axes.index(ax)
            bcs = tuple(self.boundary_recipes[key] for key in bv)
            for side, bc in zip(("left", "right"), bcs):
                side = cast(Literal["left", "right"], side)
                active_index: int = 1 if side == "left" else -2
                same_side_active_layer_idx = [slice(None)] * self.grid.ndim
                same_side_active_layer_idx[
                    iax
                ] = active_index  # type:ignore [call-overload]

                same_side_ghost_layer_idx = [slice(None)] * self.grid.ndim
                # f(-2)=-1, f(1)=0
                same_side_ghost_layer_idx[iax] = -(  # type:ignore [call-overload]
                    (active_index + 1) % 2
                )

                opposite_side_active_layer_idx = [slice(None)] * self.grid.ndim
                # f(-2)=1, f(1)=-2
                opposite_side_active_layer_idx[iax] = (  # type:ignore [call-overload]
                    1 if active_index == -2 else -2
                )

                opposite_side_ghost_layer_idx = [slice(None)] * self.grid.ndim
                # f(-2)=0, f(1)=-1
                opposite_side_ghost_layer_idx[iax] = -(  # type:ignore [call-overload]
                    active_index % 2
                )

                if weight_array is None:
                    ONE = np.ones(1, dtype=array.dtype)
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
