from ctypes import POINTER, CFUNCTYPE, c_double, c_int, c_bool, Structure, CDLL
import numpy as np


class Sp(Structure):
    _fields_ = [(name, t) for t, names in [
        (c_double, 'eps, sigma, rc, L, dt'),
        (c_int, 'N, Nt, thermo, seed'),
        (c_bool, 'dump, use_numba, use_cython, use_fortran, use_cfortran')
    ] for name in names.split(', ')]


Matrix = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F')
MatrixFunc = CFUNCTYPE(None, POINTER(c_double), POINTER(c_double), c_int)


@MatrixFunc
def inv(A_in, A_out, n):
    A_in = np.ctypeslib.as_array(A_in, (n, n))
    A_out = np.ctypeslib.as_array(A_out, (n, n))
    A_out[:] = np.linalg.inv(A_in)


ljlib = CDLL('ljcf.so')
ljlib.tot_pe.restype = c_double
ljlib.tot_pe.argtypes = [Matrix, Sp, c_int]
ljlib.force_list.restype = None
ljlib.force_list.argtypes = [Matrix, Sp, MatrixFunc, Matrix, c_int]


def tot_PE(pos_list, sp):
    return ljlib.tot_pe(np.asfortranarray(pos_list), Sp(**sp), pos_list.shape[0])


def force_list(pos_list, sp):
    F = np.zeros_like(pos_list, order='F')
    ljlib.force_list(np.asfortranarray(pos_list), Sp(**sp), inv, F, pos_list.shape[0])
    return F
