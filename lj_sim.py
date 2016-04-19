#!/usr/bin/env python3
"""
Simulation of LJ clusters in a box w pbc

Usage: lj_sim.py <L> <rho> [--T <T>] [--Nt <Nt>] [--dt <dt>] 
                 [--thermo <th>] [--dump]

Options:
    --T <T>             Temperature [default: 1.0]
    --Nt <Nt>           Number of time steps [default: 100]
    --dt <dt>           Timestep [default: 0.002]
    --thermo <th>       Print output this many times [default: 10]

02/04/16
"""
import numpy as np
import os, sys
from docopt import docopt
from lj_functions import *
from lj_io import *


class mydict(dict):
    """A contained for all the system constants"""
    def __getattr__(self, key):
        return self[key]


if __name__ == "__main__":
    args = docopt(__doc__)
#    print(args)
    L = float(args["<L>"])
    rho = float(args["<rho>"])
    N = int(rho*L**3)
    T = float(args["--T"])
    Nt = int(args["--Nt"])
    dt = float(args["--dt"])
    thermo = int(args["--thermo"])
    
    eps = 1.0
    sigma = 1.0
    rc = 3.0
    seed = 123

    if N == 0:
        print("No particles, aborting.")
        sys.exit()
    
    sp = mydict(eps=eps, sigma=sigma, rc=rc, N=N, L=L, dt=dt, Nt=Nt, \
                thermo=thermo, seed=seed, dump=args["--dump"]) # system params

    print(" =========== \n LJ clusters \n ===========")
    print("Particles: %i | Temp: %f | Steps: %i | dt: %f | thermo: %i" \
          % (N, T, Nt, dt, thermo))


    if args["--dump"]:
        dumpdir = "Dump"
        if not os.path.exists(dumpdir):
            os.makedirs(dumpdir)
 

    # init system
    print("Initialising the system...")
    pos_list, count, E = init_pos(N, sp)
    print("Number of trials: %i", count)
    vel_list = init_vel(N, T)

    # How to equilibrate?

    # run system
    print("Starting integration...")
#    xyz_frames, E = integrate(pos_list, vel_list, sp)
    E = integrate(pos_list, vel_list, sp)

    # print into file
#    Nf = xyz_frames.shape[-1]
#        for i in range(Nf):
#            fname = "Dump/dump_" + str((i+1)*thermo) + ".xyz"
#            save_xyzmatrix(fname, xyz_frames[:, :, i])

    



