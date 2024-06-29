import numpy as np
from numpy.typing import NDArray

NDArrayReal = NDArray[np.float64 | np.float32]
NDArrayUInt = NDArray[np.uint16]

def _index_particles(
    cell_edges_x1: NDArrayReal,
    cell_edges_x2: NDArrayReal,
    cell_edges_x3: NDArrayReal,
    particles_x1: NDArrayReal,
    particles_x2: NDArrayReal,
    particles_x3: NDArrayReal,
    dx: NDArrayReal,
    out: NDArrayUInt,
) -> None: ...
def _deposit_ngp_1D(
    cell_edges_x1: NDArrayReal,
    cell_edges_x2: NDArrayReal,
    cell_edges_x3: NDArrayReal,
    particles_x1: NDArrayReal,
    particles_x2: NDArrayReal,
    particles_x3: NDArrayReal,
    field: NDArrayReal,
    weight_field: NDArrayReal,
    hci: NDArrayUInt,  # dim: 2
    out: NDArrayReal,
) -> None: ...
def _deposit_ngp_2D(
    cell_edges_x1: NDArrayReal,
    cell_edges_x2: NDArrayReal,
    cell_edges_x3: NDArrayReal,
    particles_x1: NDArrayReal,
    particles_x2: NDArrayReal,
    particles_x3: NDArrayReal,
    field: NDArrayReal,
    weight_field: NDArrayReal,
    hci: NDArrayUInt,  # dim: 2
    out: NDArrayReal,
) -> None: ...
def _deposit_ngp_3D(
    cell_edges_x1: NDArrayReal,
    cell_edges_x2: NDArrayReal,
    cell_edges_x3: NDArrayReal,
    particles_x1: NDArrayReal,
    particles_x2: NDArrayReal,
    particles_x3: NDArrayReal,
    field: NDArrayReal,
    weight_field: NDArrayReal,
    hci: NDArrayUInt,  # dim: 2
    out: NDArrayReal,
) -> None: ...
def _deposit_cic_1D(
    cell_edges_x1: NDArrayReal,
    cell_edges_x2: NDArrayReal,
    cell_edges_x3: NDArrayReal,
    particles_x1: NDArrayReal,
    particles_x2: NDArrayReal,
    particles_x3: NDArrayReal,
    field: NDArrayReal,
    weight_field: NDArrayReal,
    hci: NDArrayUInt,  # dim: 2
    out: NDArrayReal,
) -> None: ...
def _deposit_cic_2D(
    cell_edges_x1: NDArrayReal,
    cell_edges_x2: NDArrayReal,
    cell_edges_x3: NDArrayReal,
    particles_x1: NDArrayReal,
    particles_x2: NDArrayReal,
    particles_x3: NDArrayReal,
    field: NDArrayReal,
    weight_field: NDArrayReal,
    hci: NDArrayUInt,  # dim: 2
    out: NDArrayReal,
) -> None: ...
def _deposit_cic_3D(
    cell_edges_x1: NDArrayReal,
    cell_edges_x2: NDArrayReal,
    cell_edges_x3: NDArrayReal,
    particles_x1: NDArrayReal,
    particles_x2: NDArrayReal,
    particles_x3: NDArrayReal,
    field: NDArrayReal,
    weight_field: NDArrayReal,
    hci: NDArrayUInt,  # dim: 2
    out: NDArrayReal,
) -> None: ...
def _deposit_tsc_1D(
    cell_edges_x1: NDArrayReal,
    cell_edges_x2: NDArrayReal,
    cell_edges_x3: NDArrayReal,
    particles_x1: NDArrayReal,
    particles_x2: NDArrayReal,
    particles_x3: NDArrayReal,
    field: NDArrayReal,
    weight_field: NDArrayReal,
    hci: NDArrayUInt,  # dim: 2
    out: NDArrayReal,
) -> None: ...
def _deposit_tsc_2D(
    cell_edges_x1: NDArrayReal,
    cell_edges_x2: NDArrayReal,
    cell_edges_x3: NDArrayReal,
    particles_x1: NDArrayReal,
    particles_x2: NDArrayReal,
    particles_x3: NDArrayReal,
    field: NDArrayReal,
    weight_field: NDArrayReal,
    hci: NDArrayUInt,  # dim: 2
    out: NDArrayReal,
) -> None: ...
def _deposit_tsc_3D(
    cell_edges_x1: NDArrayReal,
    cell_edges_x2: NDArrayReal,
    cell_edges_x3: NDArrayReal,
    particles_x1: NDArrayReal,
    particles_x2: NDArrayReal,
    particles_x3: NDArrayReal,
    field: NDArrayReal,
    weight_field: NDArrayReal,
    hci: NDArrayUInt,  # dim: 2
    out: NDArrayReal,
) -> None: ...
