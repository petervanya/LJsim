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


def V_LJ(mag_r, sp):
    '''Lennard-Jones potential, mag_r is a number.
    * sp: system params
    >>> from numpy.linalg import norm
    >>> V_LJ(norm([0.5, 0.5, 0.5]))
    12.99
    '''
    V_rc = 4 * sp.eps * ((sp.sigma / sp.rc) ** 12 - (sp.sigma / sp.rc) ** 6)
    return 4 * sp.eps * ((sp.sigma / mag_r) ** 12 - (sp.sigma / mag_r) ** 6) - \
    V_rc if mag_r < sp.rc else 0.0


def force(r, sp):
    '''r is a vector'''
    mag_dr = norm(r)
    return 4 * sp.eps * (-12 * (sp.sigma / mag_dr) ** 12 + 6 * (sp.sigma / mag_dr) ** 6) * r / mag_dr **2 \
           if mag_dr < sp.rc else np.zeros(3)


def tot_PE(pos_list, sp):
    """ MAKE THIS MORE EFFICIENT """
    E = 0.0
    N = pos_list.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            E += V_LJ(norm(pos_list[i] - pos_list[j]), sp)
    return E


def tot_KE(vel_list):
    '''Total kinetic energy of the system,
    same mass assumed'''
    return np.sum(vel_list * vel_list) / 2


def init_pos(N, sp):
    np.random.seed(sp.seed)
    E_cut = 1e5
    E = E_cut * 2
    count = 0
    while E > E_cut:
        pos_list = np.random.rand(N, 3) * sp.L
        E = tot_PE(pos_list, sp)
        count += 1
    return pos_list, count, E


def init_vel(N, T):
    '''Initialise velocities'''
    return np.random.randn(N, 3) * 3 * T


def force_list(pos_list, sp):
    '''Force matrix'''
    N = pos_list.shape[0]
    force_mat = np.zeros((N, N, 3))
    shift_vecs = sp.L * np.vstack((np.eye(3), -np.eye(3))) # FIX THIS!
    cell = sp.L*np.eye(3)
    inv_cell = np.linalg.pinv(cell)
    for i in range(N):
        for j in range(i):
            dr = pos_list[j] - pos_list[i]
            G = np.dot(inv_cell, dr)
            G_n = G - np.round(G)
            dr_n = np.dot(cell, G_n)
            force_mat[i, j] = force(dr_n, sp)
    
    force_mat -= np.transpose(force_mat, (1, 0, 2))
    return np.sum(force_mat, axis=1)


def verlet_step(pos_list2, pos_list1, sp):
    '''Verlet algorithm, returing updated position list
    and number of passes through walls'''
    F = force_list(pos_list2, sp)
    pos_list3 = (2 * pos_list2 - pos_list1) + F * sp.dt ** 2
    Npasses = np.sum(pos_list3 - pos_list3 % sp.L != 0, axis=1)
    return new_list % sp.L, Npasses


def vel_verlet_step(pos_list, vel_list, sp):
    '''The velocity Verlet algorithm, 
    returning position and velocity matrices'''
    F = force_list(pos_list, sp)
    pos_list2 = pos_list + vel_list * sp.dt + F * sp.dt**2 / 2
    F2 = force_list(pos_list2, sp)
    vel_list2 = vel_list + (F + F2) * sp.dt / 2
    Npasses = np.sum(pos_list2 - pos_list2 % sp.L != 0, axis=1)
    pos_list2 = pos_list2 % sp.L
    return pos_list2, vel_list2, Npasses


def integrate(pos_list, vel_list, sp):
    '''
    Verlet integration for Nt steps.
    Save each thermo-multiple step into xyz_frames.
    Mass set to 1.
    THINK ABOUT:
    * should enable choosing between two Verlets?
    '''
    N = pos_list.shape[0]
    Nframes = int(sp.Nt // sp.thermo)
    n_fr = 1
    xyz_frames = np.zeros((N, 3, Nframes))
    E = np.zeros(sp.Nt)

    # 1st Verlet step
    F = force_list(pos_list, sp)
    pos_list = pos_list + vel_list * sp.dt + F * sp.dt**2 / 2
    E[0] = tot_KE(vel_list) + tot_PE(pos_list, sp)

    # Other steps
    for i in range(1, sp.Nt):
        (pos_list, vel_list, Npasses) = vel_verlet_step(pos_list, vel_list, sp)
        E[i] = tot_KE(vel_list) + tot_PE(pos_list, sp)
        if i % sp.thermo == 0:
            xyz_frames[:, :, n_fr] = pos_list
            print('Step: %i, Energy: %f' % (i, E[i]))
            n_fr += 1
            continue
    return xyz_frames, E


