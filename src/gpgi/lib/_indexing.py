def _index_particles(
    cell_edges_x1,
    cell_edges_x2,
    cell_edges_x3,
    particles_x1,
    particles_x2,
    particles_x3,
    dx,
    out,
):
    if cell_edges_x3.shape[0] > 1:
        ndim = 3
    elif cell_edges_x2.shape[0] > 1:
        ndim = 2
    else:
        ndim = 1

    particle_count = particles_x1.shape[0]

    for ipart in range(particle_count):
        x = particles_x1[ipart]
        if dx[0] > 0:
            out[ipart, 0] = int((x - cell_edges_x1[0]) // dx[0])
        else:
            iL = 1
            iR = cell_edges_x1.shape[0] - 2
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

        x = particles_x2[ipart]
        if dx[1] > 0:
            out[ipart, 1] = int((x - cell_edges_x2[0]) // dx[1])
        else:
            iL = 1
            iR = cell_edges_x2.shape[00] - 2
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

        x = particles_x3[ipart]
        if dx[2] > 0:
            out[ipart, 2] = int((x - cell_edges_x3[0]) // dx[2])
        else:
            iL = 1
            iR = cell_edges_x3.shape[0] - 2
            idx = (iL + iR) // 2
            while idx != iL:
                if cell_edges_x3[idx] > x:
                    iR = idx
                else:
                    iL = idx
                idx = (iL + iR) // 2
            out[ipart, 2] = idx
