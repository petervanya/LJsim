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

    if N == 0:
        print("No particles, aborting.")
        sys.exit()

    print(" =========== \n LJ clusters \n ===========")
    print("Particles: %i | Temp: %f | Steps: %i | dt: %f | thermo: %i" \
          % (N, T, Nt, dt, thermo))

    # init system
    print("Initialising the system...")
    pos_list, count, E = init_pos(N, L)
    vel_list = init_vel(N, T)

    # run system
    print("Starting integration...")
    xyz_frames, E = integrate(pos_list, vel_list, dt, Nt, thermo, L)

    # print into file
    print(xyz_frames[:, :, -1])
    Nf = xyz_frames.shape[-1]

    if args["--dump"]:
        print("Dumping xyz files...")
        dumpdir = "Dump"
        if not os.path.exists(dumpdir):
            os.makedirs(dumpdir)
 
        for i in range(Nf):
            fname = "Dump/dump_" + str((i+1)*thermo) + ".out"
            np.savetxt(fname, xyz_frames[:, :, i])

    



