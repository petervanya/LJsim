#!/usr/bin/env python3
'''
Forces modul for LJ clusters

Questions:
* how to best pass eps, sigma, rc, L to all the functions? Should use *args, or **kwargs?

31/03/16
'''
import numpy as np
from numpy.linalg import norm

sysparams = [1, 1, 3, 10]

class Sysparams(object):
    def __init__(self):
        self.eps = 1
        self.sigma = 1
        self.rc = 1
        self.L = 10


def V_LJ(mag_r, eps = 1, sigma = 1, rc = 3, *args):
    '''Lennard-Jones potential, mag_r is a number.
    >>> from numpy.linalg import norm
    >>> V_LJ(norm([0.5, 0.5, 0.5]))
    12.99
    '''
    V_rc = 4 * eps * ((sigma / rc) ** 12 - (sigma / rc) ** 6)
    if mag_r < rc:
        return 4 * eps * ((sigma / mag_r) ** 12 - (sigma / mag_r) ** 6) - V_rc


def force(r, eps = 1, sigma = 1, rc = 3, *args):
    '''r is a vector'''
    mag_dr = norm(r)
    if mag_dr < rc:
        return 4 * eps * (-12 * (sigma / mag_dr) ** 12 + 6 * (sigma / mag_dr) ** 6) * r / mag_dr ** 2
    return np.zeros(3)


def tot_PE(pos_list, *sysparams):
    n_fr = 0
    E = 0
    N = pos_list.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            E += V_LJ(norm(pos_list[i] - pos_list[j]), *sysparams)
    return E


def tot_KE(vel_list):
    '''Total kinetic energy of the system,
    same mass assumed'''
    return np.sum(vel_list * vel_list) / 2


def init_pos(N, L, seed = 123, *sysparams):
    np.random.seed(seed)
    E_cut = 100000
    E = E_cut * 2
    count = 0
    while E > E_cut:
        pos_list = np.random.rand(N, 3) * L
        E = tot_PE(pos_list, *sysparams)
        count += 1
    return pos_list, count, E


def init_vel(N, T):
    '''Initialise velocities'''
    return np.random.randn(N, 3) * 3 * T


def force_list(pos_list, L, *sysparams):
    '''Force matrix'''
    N = pos_list.shape[0]
    force_mat = np.zeros((N, N, 3))
    shift_vecs = L * np.vstack((np.eye(3), -np.eye(3)))
    for i in range(N):
        for j in range(i):
            dr = pos_list[j] - pos_list[i]
            for dirn in range(6):
                dr_new = pos_list[i] - pos_list[j] + shift_vecs[dirn]
                if norm(dr_new) < norm(dr):
                    dr = dr_new
                    continue
            force_mat[(i, j)] = force(dr, *sysparams)
        
    
    force_mat -= np.transpose(force_mat, (1, 0, 2))
    return np.sum(force_mat, axis=1)


def verlet_step(pos_list2, pos_list1, dt, L, *sysparams):
    '''Verlet algorithm, returing updated position list
    and number of passes through walls'''
    F = force_list(pos_list2, L, *sysparams)
    pos_list3 = (2 * pos_list2 - pos_list1) + force * dt ** 2
    Npasses = np.sum(pos_list3 - pos_list3 % L != 0, axis = 1)
    return new_list % L, Npasses


def vel_verlet_step(pos_list, vel_list, dt, L, *sysparams):
    '''The velocity Verlet algorithm, 
    returning position and velocity matrices'''
    F = force_list(pos_list, L, *sysparams)
    pos_list2 = pos_list + vel_list * dt + F * dt ** 2 / 2
    F2 = force_list(pos_list2, L, *sysparams)
    vel_list2 = vel_list + (F + F2) * dt / 2
    Npasses = np.sum(pos_list2 - pos_list2 % L != 0, axis = 1)
    pos_list2 = pos_list2 % L
    return pos_list2, vel_list2, Npasses


def integrate(pos_list, vel_list, dt, Nt, thermo, L, *sysparams):
    '''
    Verlet integration for Nt steps.
    Save each thermo-multiple step into xyz_frames
    mass set to 1
    To think about:
    * should enable choosing between two Verlets?
    '''
    N = pos_list.shape[0]
    Nframes = int(Nt // thermo)
    n_fr = 1
    xyz_frames = np.zeros((N, 3, Nframes))
    E = np.zeros(Nt)
    F = force_list(pos_list, L, *sysparams)
    pos_list = pos_list + vel_list * dt + F * dt ** 2 / 2
    E[0] = tot_KE(vel_list) + tot_PE(pos_list, *sysparams)
    for i in range(1, Nt):
        (pos_list, vel_list, Npasses) = vel_verlet_step(pos_list, vel_list, dt, L, *sysparams)
        E[i] = tot_KE(vel_list) + tot_PE(pos_list, *sysparams)
        if i % thermo == 0:
            xyz_frames[(:, :, n_fr)] = pos_list
            print('Step: %i, Energy: %f' % (i, E[i]))
            n_fr += 1
            continue
    return xyz_frames, E

