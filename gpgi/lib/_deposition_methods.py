def _deposit_pic(
    cell_edges_x1, cell_edges_x2, cell_edges_x3, particle_coords, field, hci, out
):
    for ipart in range(len(hci)):
        md_idx = tuple(hci[ipart])
        out[md_idx] += field[ipart]
