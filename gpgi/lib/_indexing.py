def _index_particles(
    ndim,
    cell_edges_x1,
    cell_edges_x2,
    cell_edges_x3,
    particle_count,
    particle_coords,
    out,
):
    for ipart in range(particle_count):
        x = particle_coords[ipart, 0]
        edges = cell_edges_x1
        max_idx = len(cell_edges_x1) - 1
        idx = 0
        while idx < max_idx and edges[idx + 1] < x:
            idx += 1
        out[ipart, 0] = idx

        if ndim < 2:
            continue

        x = particle_coords[ipart, 1]
        edges = cell_edges_x2
        max_idx = len(cell_edges_x2) + 1
        idx = 0
        while idx < max_idx and edges[idx + 1] < x:
            idx += 1
        out[ipart, 1] = idx

        if ndim < 3:
            continue

        x = particle_coords[ipart, 2]
        edges = cell_edges_x3
        max_idx = len(cell_edges_x3) + 1
        idx = 0
        while idx < max_idx and edges[idx + 1] < x:
            idx += 1
        out[ipart, 2] = idx
