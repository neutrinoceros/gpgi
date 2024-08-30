# cython: freethreading_compatible = True
# note that this macro is defined only to prevent the interpreter from
# re-enabling the GIL when this extension is imported from, but the functions
# themselves are absolutely not thread-safe since they work by mutating objects
# that can be accessed from another thread (namely, the `out` argument).
# This design choice was made to remove the overhead of memory allocation in
# performance measurements. It is fine to keep this interface as long as the out
# argument isn't directly exposed in public API.

cimport cython
cimport numpy as np
from libc.math cimport floor, fmin

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

    # reduce the chance of floating point arithmetic errors
    # by forcing positions be read as double precision
    # (instead of reading as the input type)
    cdef np.float64_t x

    cdef int ndim
    if cell_edges_x3.shape[0] > 1:
        ndim = 3
    elif cell_edges_x2.shape[0] > 1:
        ndim = 2
    else:
        ndim = 1

    particle_count = particles_x1.shape[0]

    with nogil:
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


@cython.boundscheck(False)
@cython.wraparound(False)
def _deposit_ngp_1D(
    np.ndarray[real, ndim=1] cell_edges_x1,
    np.ndarray[real, ndim=1] cell_edges_x2,
    np.ndarray[real, ndim=1] cell_edges_x3,
    np.ndarray[real, ndim=1] particles_x1,
    np.ndarray[real, ndim=1] particles_x2,
    np.ndarray[real, ndim=1] particles_x3,
    np.ndarray[real, ndim=1] field,
    np.ndarray[real, ndim=1] weight_field,
    np.ndarray[np.uint16_t, ndim=2] hci,
    np.ndarray[real, ndim=1] out,
):
    cdef Py_ssize_t ipart, particle_count
    particle_count = hci.shape[0]

    cdef int i

    cdef np.uint16_t[:, :] hci_v = hci
    cdef real[:] field_v = field
    cdef real[:] out_v = out

    cdef int lw = weight_field.shape[0]
    cdef real[:] wfield_v = weight_field
    cdef bint no_weight = (lw == 0)

    with nogil:
        for ipart in range(particle_count):
            i = hci_v[ipart][0]
            if no_weight:
                out_v[i] += field_v[ipart]
            else:
                out_v[i] += field_v[ipart] * wfield_v[ipart]


@cython.boundscheck(False)
@cython.wraparound(False)
def _deposit_ngp_2D(
    np.ndarray[real, ndim=1] cell_edges_x1,
    np.ndarray[real, ndim=1] cell_edges_x2,
    np.ndarray[real, ndim=1] cell_edges_x3,
    np.ndarray[real, ndim=1] particles_x1,
    np.ndarray[real, ndim=1] particles_x2,
    np.ndarray[real, ndim=1] particles_x3,
    np.ndarray[real, ndim=1] field,
    np.ndarray[real, ndim=1] weight_field,
    np.ndarray[np.uint16_t, ndim=2] hci,
    np.ndarray[real, ndim=2] out,
):
    cdef Py_ssize_t ipart, particle_count
    particle_count = hci.shape[0]

    cdef int i, j

    cdef np.uint16_t[:, :] hci_v = hci
    cdef real[:] field_v = field
    cdef real[:, :] out_v = out

    cdef int lw = weight_field.shape[0]
    cdef real[:] wfield_v = weight_field
    cdef bint no_weight = (lw == 0)

    with nogil:
        for ipart in range(particle_count):
            i = hci_v[ipart][0]
            j = hci_v[ipart][1]
            if no_weight:
                out_v[i][j] += field_v[ipart]
            else:
                out_v[i][j] += field_v[ipart] * wfield_v[ipart]


@cython.boundscheck(False)
@cython.wraparound(False)
def _deposit_ngp_3D(
    np.ndarray[real, ndim=1] cell_edges_x1,
    np.ndarray[real, ndim=1] cell_edges_x2,
    np.ndarray[real, ndim=1] cell_edges_x3,
    np.ndarray[real, ndim=1] particles_x1,
    np.ndarray[real, ndim=1] particles_x2,
    np.ndarray[real, ndim=1] particles_x3,
    np.ndarray[real, ndim=1] field,
    np.ndarray[real, ndim=1] weight_field,
    np.ndarray[np.uint16_t, ndim=2] hci,
    np.ndarray[real, ndim=3] out,
):
    cdef Py_ssize_t ipart, particle_count
    particle_count = hci.shape[0]

    cdef int i, j, k

    cdef np.uint16_t[:, :] hci_v = hci
    cdef real[:] field_v = field
    cdef real[:, :, :] out_v = out

    cdef int lw = weight_field.shape[0]
    cdef real[:] wfield_v = weight_field
    cdef bint no_weight = (lw == 0)

    with nogil:
        for ipart in range(particle_count):
            i = hci_v[ipart][0]
            j = hci_v[ipart][1]
            k = hci_v[ipart][2]
            if no_weight:
                out_v[i][j][k] += field_v[ipart]
            else:
                out_v[i][j][k] += field_v[ipart] * wfield_v[ipart]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _deposit_cic_1D(
    np.ndarray[real, ndim=1] cell_edges_x1,
    np.ndarray[real, ndim=1] cell_edges_x2,
    np.ndarray[real, ndim=1] cell_edges_x3,
    np.ndarray[real, ndim=1] particles_x1,
    np.ndarray[real, ndim=1] particles_x2,
    np.ndarray[real, ndim=1] particles_x3,
    np.ndarray[real, ndim=1] field,
    np.ndarray[real, ndim=1] weight_field,
    np.ndarray[np.uint16_t, ndim=2] hci,
    np.ndarray[real, ndim=1] out,
):

    cdef Py_ssize_t ipart, particle_count, i, oci
    particle_count = hci.shape[0]

    cdef real x, d
    cdef np.uint16_t ci, ci_start

    cdef real[:] field_v = field
    cdef real[:] out_v = out

    # geometric weight arrays
    cdef real[2] w

    cdef int lw = weight_field.shape[0]
    cdef real[:] wfield_v = weight_field
    cdef bint no_weight = (lw == 0)

    with nogil:
        for ipart in range(particle_count):
            x = particles_x1[ipart]
            ci = hci[ipart, 0]
            d = (x - cell_edges_x1[ci]) / (cell_edges_x1[ci + 1] - cell_edges_x1[ci])
            ci_start = ci + <np.uint16_t>fmin(0, floor(2*d)-1)
            w[0] = 0.5 - d + fmin(1, floor(2*d))
            w[1] = 1 - w[0]

            for i in range(2):
                oci = ci_start + i
                if no_weight:
                    out_v[oci] += w[i] * field_v[ipart]
                else:
                    out_v[oci] += w[i] * field_v[ipart] * wfield_v[ipart]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _deposit_cic_2D(
    np.ndarray[real, ndim=1] cell_edges_x1,
    np.ndarray[real, ndim=1] cell_edges_x2,
    np.ndarray[real, ndim=1] cell_edges_x3,
    np.ndarray[real, ndim=1] particles_x1,
    np.ndarray[real, ndim=1] particles_x2,
    np.ndarray[real, ndim=1] particles_x3,
    np.ndarray[real, ndim=1] field,
    np.ndarray[real, ndim=1] weight_field,
    np.ndarray[np.uint16_t, ndim=2] hci,
    np.ndarray[real, ndim=2] out,
):

    cdef Py_ssize_t ipart, particle_count, i, j, oci, ocj
    particle_count = hci.shape[0]

    cdef real x, d
    cdef np.uint16_t ci, cj, ci_start, cj_start

    cdef real[:] field_v = field
    cdef real[:, :] out_v = out

    # geometric weight arrays
    cdef real[2] w1, w2
    cdef real[2][2] w

    cdef int lw = weight_field.shape[0]
    cdef real[:] wfield_v = weight_field
    cdef bint no_weight = (lw == 0)

    with nogil:
        for ipart in range(particle_count):
            x = particles_x1[ipart]
            ci = hci[ipart, 0]
            d = (x - cell_edges_x1[ci]) / (cell_edges_x1[ci + 1] - cell_edges_x1[ci])
            ci_start = ci + <np.uint16_t>fmin(0, floor(2*d)-1)
            w1[0] = 0.5 - d + fmin(1, floor(2*d))
            w1[1] = 1 - w1[0]

            x = particles_x2[ipart]
            cj = hci[ipart, 1]
            d = (x - cell_edges_x2[cj]) / (cell_edges_x2[cj + 1] - cell_edges_x2[cj])
            cj_start = cj + <np.uint16_t>fmin(0, floor(2*d)-1)
            w2[0] = 0.5 - d + fmin(1, floor(2*d))
            w2[1] = 1 - w2[0]

            for i in range(2):
                for j in range(2):
                    w[i][j] = w1[i] * w2[j]

            for i in range(2):
                oci = ci_start + i
                for j in range(2):
                    ocj = cj_start + j
                    if no_weight:
                        out_v[oci, ocj] += w[i][j] * field_v[ipart]
                    else:
                        out_v[oci, ocj] += w[i][j] * field_v[ipart] * wfield_v[ipart]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _deposit_cic_3D(
    np.ndarray[real, ndim=1] cell_edges_x1,
    np.ndarray[real, ndim=1] cell_edges_x2,
    np.ndarray[real, ndim=1] cell_edges_x3,
    np.ndarray[real, ndim=1] particles_x1,
    np.ndarray[real, ndim=1] particles_x2,
    np.ndarray[real, ndim=1] particles_x3,
    np.ndarray[real, ndim=1] field,
    np.ndarray[real, ndim=1] weight_field,
    np.ndarray[np.uint16_t, ndim=2] hci,
    np.ndarray[real, ndim=3] out,
):

    cdef Py_ssize_t ipart, particle_count, i, j, k, oci, ocj, ock
    particle_count = hci.shape[0]

    cdef real x, d
    cdef np.uint16_t ci, cj, ck, ci_start, cj_start, ck_start

    cdef real[:] field_v = field
    cdef real[:, :, :] out_v = out

    # geometric weight arrays
    cdef real[2] w1, w2, w3
    cdef real[2][2][2] w

    cdef int lw = weight_field.shape[0]
    cdef real[:] wfield_v = weight_field
    cdef bint no_weight = (lw == 0)

    with nogil:
        for ipart in range(particle_count):
            x = particles_x1[ipart]
            ci = hci[ipart, 0]
            d = (x - cell_edges_x1[ci]) / (cell_edges_x1[ci + 1] - cell_edges_x1[ci])
            ci_start = ci + <np.uint16_t>fmin(0, floor(2*d)-1)
            w1[0] = 0.5 - d + fmin(1, floor(2*d))
            w1[1] = 1 - w1[0]

            x = particles_x2[ipart]
            cj = hci[ipart, 1]
            d = (x - cell_edges_x2[cj]) / (cell_edges_x2[cj + 1] - cell_edges_x2[cj])
            cj_start = cj + <np.uint16_t>fmin(0, floor(2*d)-1)
            w2[0] = 0.5 - d + fmin(1, floor(2*d))
            w2[1] = 1 - w2[0]

            x = particles_x3[ipart]
            ck = hci[ipart, 2]
            d = (x - cell_edges_x3[ck]) / (cell_edges_x3[ck + 1] - cell_edges_x3[ck])
            ck_start = ck + <np.uint16_t>fmin(0, floor(2*d)-1)
            w3[0] = 0.5 - d + fmin(1, floor(2*d))
            w3[1] = 1 - w3[0]

            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        w[i][j][k] = w1[i] * w2[j] * w3[k]

            for i in range(2):
                oci = ci_start + i
                for j in range(2):
                    ocj = cj_start + j
                    for k in range(2):
                        ock = ck_start + k
                        if no_weight:
                            out_v[oci, ocj, ock] += w[i][j][k] * field_v[ipart]
                        else:
                            out_v[oci, ocj, ock] += w[i][j][k] * field_v[ipart] * wfield_v[ipart]  # noqa E501


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _deposit_tsc_1D(
    np.ndarray[real, ndim=1] cell_edges_x1,
    np.ndarray[real, ndim=1] cell_edges_x2,
    np.ndarray[real, ndim=1] cell_edges_x3,
    np.ndarray[real, ndim=1] particles_x1,
    np.ndarray[real, ndim=1] particles_x2,
    np.ndarray[real, ndim=1] particles_x3,
    np.ndarray[real, ndim=1] field,
    np.ndarray[real, ndim=1] weight_field,
    np.ndarray[np.uint16_t, ndim=2] hci,
    np.ndarray[real, ndim=1] out,
):
    cdef Py_ssize_t ipart, particle_count, i, oci
    particle_count = hci.shape[0]

    cdef real x, d
    cdef np.uint16_t ci

    cdef real[:] field_v = field
    cdef real[:] out_v = out

    # geometric weight arrays
    cdef real[3] w

    cdef int lw = weight_field.shape[0]
    cdef real[:] wfield_v = weight_field
    cdef bint no_weight = (lw == 0)

    with nogil:
        for ipart in range(particle_count):
            x = particles_x1[ipart]
            ci = hci[ipart, 0]
            d = (x - cell_edges_x1[ci]) / (cell_edges_x1[ci + 1] - cell_edges_x1[ci])
            w[0] = 0.5 * (1 - d) ** 2
            w[1] = 0.75 - (d - 0.5) ** 2
            w[2] = 0.5 * d**2

            for i in range(3):
                oci = ci - 1 + i
                if no_weight:
                    out_v[oci] += w[i] * field_v[ipart]
                else:
                    out_v[oci] += w[i] * field_v[ipart] * wfield_v[ipart]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _deposit_tsc_2D(
    np.ndarray[real, ndim=1] cell_edges_x1,
    np.ndarray[real, ndim=1] cell_edges_x2,
    np.ndarray[real, ndim=1] cell_edges_x3,
    np.ndarray[real, ndim=1] particles_x1,
    np.ndarray[real, ndim=1] particles_x2,
    np.ndarray[real, ndim=1] particles_x3,
    np.ndarray[real, ndim=1] field,
    np.ndarray[real, ndim=1] weight_field,
    np.ndarray[np.uint16_t, ndim=2] hci,
    np.ndarray[real, ndim=2] out,
):
    cdef Py_ssize_t ipart, particle_count, i, j, oci, ocj
    particle_count = hci.shape[0]

    cdef real x, d
    cdef np.uint16_t ci, cj

    cdef real[:] field_v = field
    cdef real[:, :] out_v = out

    # geometric weight arrays
    cdef real[3] w1, w2
    cdef real[3][3] w

    cdef int lw = weight_field.shape[0]
    cdef real[:] wfield_v = weight_field
    cdef bint no_weight = (lw == 0)

    with nogil:
        for ipart in range(particle_count):
            x = particles_x1[ipart]
            ci = hci[ipart, 0]
            d = (x - cell_edges_x1[ci]) / (cell_edges_x1[ci + 1] - cell_edges_x1[ci])
            w1[0] = 0.5 * (1 - d) ** 2
            w1[1] = 0.75 - (d - 0.5) ** 2
            w1[2] = 0.5 * d**2

            x = particles_x2[ipart]
            cj = hci[ipart, 1]
            d = (x - cell_edges_x2[cj]) / (cell_edges_x2[cj + 1] - cell_edges_x2[cj])
            w2[0] = 0.5 * (1 - d) ** 2
            w2[1] = 0.75 - (d - 0.5) ** 2
            w2[2] = 0.5 * d**2

            for i in range(3):
                for j in range(3):
                    w[i][j] = w1[i] * w2[j]

            for i in range(3):
                oci = ci - 1 + i
                for j in range(3):
                    ocj = cj - 1 + j
                    if no_weight:
                        out_v[oci, ocj] += w[i][j] * field_v[ipart]
                    else:
                        out_v[oci, ocj] += w[i][j] * field_v[ipart] * wfield_v[ipart]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _deposit_tsc_3D(
    np.ndarray[real, ndim=1] cell_edges_x1,
    np.ndarray[real, ndim=1] cell_edges_x2,
    np.ndarray[real, ndim=1] cell_edges_x3,
    np.ndarray[real, ndim=1] particles_x1,
    np.ndarray[real, ndim=1] particles_x2,
    np.ndarray[real, ndim=1] particles_x3,
    np.ndarray[real, ndim=1] field,
    np.ndarray[real, ndim=1] weight_field,
    np.ndarray[np.uint16_t, ndim=2] hci,
    np.ndarray[real, ndim=3] out,
):
    cdef Py_ssize_t ipart, particle_count, i, j, k, oci, ocj, ock
    particle_count = hci.shape[0]

    cdef real x, d
    cdef np.uint16_t ci, cj, ck

    cdef real[:] field_v = field
    cdef real[:, :, :] out_v = out

    # geometric weight arrays
    cdef real[3] w1, w2, w3
    cdef real[3][3][3] w

    cdef int lw = weight_field.shape[0]
    cdef real[:] wfield_v = weight_field
    cdef bint no_weight = (lw == 0)

    with nogil:
        for ipart in range(particle_count):
            x = particles_x1[ipart]
            ci = hci[ipart, 0]
            d = (x - cell_edges_x1[ci]) / (cell_edges_x1[ci + 1] - cell_edges_x1[ci])
            w1[0] = 0.5 * (1 - d) ** 2
            w1[1] = 0.75 - (d - 0.5) ** 2
            w1[2] = 0.5 * d**2

            x = particles_x2[ipart]
            cj = hci[ipart, 1]
            d = (x - cell_edges_x2[cj]) / (cell_edges_x2[cj + 1] - cell_edges_x2[cj])
            w2[0] = 0.5 * (1 - d) ** 2
            w2[1] = 0.75 - (d - 0.5) ** 2
            w2[2] = 0.5 * d**2

            x = particles_x3[ipart]
            ck = hci[ipart, 2]
            d = (x - cell_edges_x3[ck]) / (cell_edges_x3[ck + 1] - cell_edges_x3[ck])
            w3[0] = 0.5 * (1 - d) ** 2
            w3[1] = 0.75 - (d - 0.5) ** 2
            w3[2] = 0.5 * d**2

            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        w[i][j][k] = w1[i] * w2[j] * w3[k]

            for i in range(3):
                oci = ci - 1 + i
                for j in range(3):
                    ocj = cj - 1 + j
                    for k in range(3):
                        ock = ck - 1 + k
                        if no_weight:
                            out_v[oci, ocj, ock] += w[i][j][k] * field_v[ipart]
                        else:
                            out_v[oci, ocj, ock] += w[i][j][k] * field_v[ipart] * wfield_v[ipart]  # noqa E501
