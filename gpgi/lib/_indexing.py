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
        iL = 0
        iR = cell_edges_x1.shape[0] - 1
        idx = (iL + iR) // 2
        while idx != iL:
            if cell_edges_x1[idx] > x:
                iR = idx
            else:
                iL = idx
            idx = (iL + iR) // 2
        out[ipart, 0] = idx

        if ndim < 2:
            continue

        x = particle_coords[ipart, 1]
        iL = 0
        iR = cell_edges_x2.shape[0] - 1
        idx = (iL + iR) // 2
        while idx != iL:
            if cell_edges_x2[idx] > x:
                iR = idx
            else:
                iL = idx
            idx = (iL + iR) // 2
        out[ipart, 1] = idx

        if ndim < 3:
            continue

        x = particle_coords[ipart, 2]
        iL = 0
        iR = cell_edges_x3.shape[0] - 1
        idx = (iL + iR) // 2
        while idx != iL:
            if cell_edges_x3[idx] > x:
                iR = idx
            else:
                iL = idx
            idx = (iL + iR) // 2
        out[ipart, 2] = idx
