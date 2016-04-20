import numpy as np
cimport numpy as np
import cython
from libc.math cimport sqrt, round


cdef struct Sp:
    double eps
    double sigma
    double rc
    int N
    double L
    double dt
    int Nt
    int thermo
    int seed
    bint dump
    bint use_numba
    bint use_cython


@cython.boundscheck(False)
cdef double norm(double[:] v):
    cdef double result = 0.
    for i in range(v.shape[0]):
        result += v[i]*v[i]
    return sqrt(result)


@cython.cdivision(True)
cdef double V_LJ(double mag_r, Sp sp):
    V_rc = 4 * sp.eps * ((sp.sigma / sp.rc) ** 12 - (sp.sigma / sp.rc) ** 6)
    return 4 * sp.eps * ((sp.sigma / mag_r) ** 12 - (sp.sigma / mag_r) ** 6) - \
        V_rc if mag_r < sp.rc else 0.0


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void force(double[:] r, Sp sp, double[:] result):
    mag_dr = norm(r)
    for i in range(3):
        result[i] = 4 * sp.eps * (-12 * (sp.sigma / mag_dr) ** 12  + 6 * (sp.sigma / mag_dr) ** 6) * r[i] / mag_dr**2 \
        if mag_dr < sp.rc else 0.


@cython.boundscheck(False)
def tot_PE(double[:, :] pos_list, Sp sp):
    cdef double E = 0.
    cdef int N = pos_list.shape[0]
    cdef double[:] diff = np.zeros((3,))
    cdef int i, j, k
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(3):
                diff[k] = pos_list[i, k] - pos_list[j, k]
            E += V_LJ(norm(diff), sp)
    return E


@cython.boundscheck(False)
@cython.wraparound(False)
def force_list(double[:, :] pos_list, Sp sp):
    cdef int N = pos_list.shape[0]
    cdef double[:, :, :] force_mat = np.zeros((N, N, 3))
    cdef double[:, :] cell = sp.L*np.eye(3)
    cdef double[:, :] inv_cell = np.linalg.pinv(cell)
    cdef double[:] dr = np.zeros((3,))
    cdef double[:] dr_n = np.zeros((3,))
    cdef double[:] G = np.zeros((3,))
    cdef double[:] G_n = np.zeros((3,))
    cdef int i, j, k, l
    for i in range(N):
        for j in range(i):
            for k in range(3):
                dr[k] = pos_list[j, k] - pos_list[i, k]
            G[:] = 0.
            for k in range(3):
                for l in range(3):
                    G[k] += inv_cell[k, l]*dr[l]
            for k in range(3):
                G_n[k] = G[k] - round(G[k])
            dr_n[:] = 0.
            for k in range(3):
                for l in range(3):
                    dr_n[k] += cell[k, l]*G_n[l]
            force(dr_n, sp, force_mat[i, j, :])
    force_mat -= np.transpose(force_mat, (1, 0, 2))
    return np.sum(force_mat, axis=1)
