from gpgi._typing import D0, D1, D2, D3, F, FArray, HCIArray

def _index_particles(
    cell_edges_x1: FArray[D1, F],
    cell_edges_x2: FArray[D1, F],
    cell_edges_x3: FArray[D1, F],
    particles_x1: FArray[D1, F],
    particles_x2: FArray[D1, F],
    particles_x3: FArray[D1, F],
    dx: FArray[D1, F],
    out: HCIArray,
) -> None: ...
def _deposit_ngp_1D(
    cell_edges_x1: FArray[D1, F],
    cell_edges_x2: FArray[D1, F],
    cell_edges_x3: FArray[D1, F],
    particles_x1: FArray[D1, F],
    particles_x2: FArray[D1, F],
    particles_x3: FArray[D1, F],
    field: FArray[D1, F],
    weight_field: FArray[D1, F] | FArray[D0, F],
    hci: HCIArray,  # dim: 2
    out: FArray[D1, F],
) -> None: ...
def _deposit_ngp_2D(
    cell_edges_x1: FArray[D1, F],
    cell_edges_x2: FArray[D1, F],
    cell_edges_x3: FArray[D1, F],
    particles_x1: FArray[D1, F],
    particles_x2: FArray[D1, F],
    particles_x3: FArray[D1, F],
    field: FArray[D1, F],
    weight_field: FArray[D1, F] | FArray[D0, F],
    hci: HCIArray,  # dim: 2
    out: FArray[D2, F],
) -> None: ...
def _deposit_ngp_3D(
    cell_edges_x1: FArray[D1, F],
    cell_edges_x2: FArray[D1, F],
    cell_edges_x3: FArray[D1, F],
    particles_x1: FArray[D1, F],
    particles_x2: FArray[D1, F],
    particles_x3: FArray[D1, F],
    field: FArray[D1, F],
    weight_field: FArray[D1, F] | FArray[D0, F],
    hci: HCIArray,  # dim: 2
    out: FArray[D3, F],
) -> None: ...
def _deposit_cic_1D(
    cell_edges_x1: FArray[D1, F],
    cell_edges_x2: FArray[D1, F],
    cell_edges_x3: FArray[D1, F],
    particles_x1: FArray[D1, F],
    particles_x2: FArray[D1, F],
    particles_x3: FArray[D1, F],
    field: FArray[D1, F],
    weight_field: FArray[D1, F] | FArray[D0, F],
    hci: HCIArray,  # dim: 2
    out: FArray[D1, F],
) -> None: ...
def _deposit_cic_2D(
    cell_edges_x1: FArray[D1, F],
    cell_edges_x2: FArray[D1, F],
    cell_edges_x3: FArray[D1, F],
    particles_x1: FArray[D1, F],
    particles_x2: FArray[D1, F],
    particles_x3: FArray[D1, F],
    field: FArray[D1, F],
    weight_field: FArray[D1, F] | FArray[D0, F],
    hci: HCIArray,  # dim: 2
    out: FArray[D2, F],
) -> None: ...
def _deposit_cic_3D(
    cell_edges_x1: FArray[D1, F],
    cell_edges_x2: FArray[D1, F],
    cell_edges_x3: FArray[D1, F],
    particles_x1: FArray[D1, F],
    particles_x2: FArray[D1, F],
    particles_x3: FArray[D1, F],
    field: FArray[D1, F],
    weight_field: FArray[D1, F] | FArray[D0, F],
    hci: HCIArray,  # dim: 2
    out: FArray[D3, F],
) -> None: ...
def _deposit_tsc_1D(
    cell_edges_x1: FArray[D1, F],
    cell_edges_x2: FArray[D1, F],
    cell_edges_x3: FArray[D1, F],
    particles_x1: FArray[D1, F],
    particles_x2: FArray[D1, F],
    particles_x3: FArray[D1, F],
    field: FArray[D1, F],
    weight_field: FArray[D1, F] | FArray[D0, F],
    hci: HCIArray,  # dim: 2
    out: FArray[D1, F],
) -> None: ...
def _deposit_tsc_2D(
    cell_edges_x1: FArray[D1, F],
    cell_edges_x2: FArray[D1, F],
    cell_edges_x3: FArray[D1, F],
    particles_x1: FArray[D1, F],
    particles_x2: FArray[D1, F],
    particles_x3: FArray[D1, F],
    field: FArray[D1, F],
    weight_field: FArray[D1, F] | FArray[D0, F],
    hci: HCIArray,  # dim: 2
    out: FArray[D2, F],
) -> None: ...
def _deposit_tsc_3D(
    cell_edges_x1: FArray[D1, F],
    cell_edges_x2: FArray[D1, F],
    cell_edges_x3: FArray[D1, F],
    particles_x1: FArray[D1, F],
    particles_x2: FArray[D1, F],
    particles_x3: FArray[D1, F],
    field: FArray[D1, F],
    weight_field: FArray[D1, F] | FArray[D0, F],
    hci: HCIArray,  # dim: 2
    out: FArray[D3, F],
) -> None: ...
