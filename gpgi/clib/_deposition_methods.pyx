cimport cython
cimport numpy as np


cdef fused real:
    np.float64_t
    np.float32_t

@cython.boundscheck(False)
@cython.wraparound(False)
def _deposit_pic_1D(
    np.ndarray[real, ndim=1] field,
    np.ndarray[np.uint16_t, ndim=2] hci,
    np.ndarray[real, ndim=1] out,
):
    cdef Py_ssize_t ipart, particle_count
    particle_count = hci.shape[0]

    cdef int i

    cdef np.uint16_t[:, :] hci_v = hci
    cdef real[:] field_v = field
    cdef real[:] out_v = out

    for ipart in range(particle_count):
        i = hci_v[ipart][0]
        out_v[i] += field_v[ipart]

@cython.boundscheck(False)
@cython.wraparound(False)
def _deposit_pic_2D(
    np.ndarray[real, ndim=1] field,
    np.ndarray[np.uint16_t, ndim=2] hci,
    np.ndarray[real, ndim=2] out,
):
    cdef Py_ssize_t ipart, particle_count
    particle_count = hci.shape[0]

    cdef int i, j

    cdef np.uint16_t[:, :] hci_v = hci
    cdef real[:] field_v = field
    cdef real[:, :] out_v = out

    for ipart in range(particle_count):
        i = hci_v[ipart][0]
        j = hci_v[ipart][1]
        out_v[i][j] += field_v[ipart]


@cython.boundscheck(False)
@cython.wraparound(False)
def _deposit_pic_3D(
    np.ndarray[real, ndim=1] field,
    np.ndarray[np.uint16_t, ndim=2] hci,
    np.ndarray[real, ndim=3] out,
):
    cdef Py_ssize_t ipart, particle_count
    particle_count = hci.shape[0]

    cdef int i, j, k

    cdef np.uint16_t[:, :] hci_v = hci
    cdef real[:] field_v = field
    cdef real[:, :, :] out_v = out

    for ipart in range(particle_count):
        i = hci_v[ipart][0]
        j = hci_v[ipart][1]
        k = hci_v[ipart][2]
        out_v[i][j][k] += field_v[ipart]
