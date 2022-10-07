cimport cython
cimport numpy as np


cdef fused real:
    np.float64_t
    np.float32_t

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _index_particles(
    np.ndarray[real, ndim=1] cell_edges_x1,
    np.ndarray[real, ndim=1] cell_edges_x2,
    np.ndarray[real, ndim=1] cell_edges_x3,
    np.ndarray[real, ndim=1] particles_x1,
    np.ndarray[real, ndim=1] particles_x2,
    np.ndarray[real, ndim=1] particles_x3,
    np.ndarray[real, ndim=1] dx,
    np.ndarray[np.uint16_t, ndim=2] out,
):
    r"""
    compute the host cell index (*out*)
    The result is a mapping from particle index to containing cell index.
    Note that cell edges are to be treated as ordained from left to right
    and are padded with ghost layers (one for each side and direction).
    Particles on the other hand are not assumed to be sorted, but they *are*
    assumed to be all contained in the active domain.

    dx is a 3 element array that contains the constant step used in each direction,
    if any, or -1 as a filler value otherwise.

    In directions where a constant step is used, the result is immediate (O(1))
    Otherwise we use a bisection search (O(log(N)))
    """
    cdef Py_ssize_t ipart, particle_count
    cdef np.uint16_t iL, iR, idx
    cdef real x

    cdef int ndim
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
            iR = cell_edges_x2.shape[0] - 2
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
