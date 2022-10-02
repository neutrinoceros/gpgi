cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def _index_particles(
    int ndim,
    np.ndarray[np.float64_t, ndim=1] cell_edges_x1,
    np.ndarray[np.float64_t, ndim=1] cell_edges_x2,
    np.ndarray[np.float64_t, ndim=1] cell_edges_x3,
    int particle_count,
    np.ndarray[np.float64_t, ndim=2] particle_coords,
    np.ndarray[np.uint16_t, ndim=2] out,
):
    cdef Py_ssize_t ipart
    cdef np.uint16_t iL, iR, idx
    cdef np.float64_t x

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
