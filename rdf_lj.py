#!/usr/bin/env python3
"""
Compute RDF and save into file.

Usage:
   rdf.py <infiles> [--L <L>] [--bins <nb>]

Options:
    --L <L>             Box size [default: 10]
    --bins <nb>         Number of bins [default: 30]

03/04/16
"""
import numpy as np
import glob
from docopt import docopt
from lj_io import read_xyzmatrix


def dmatrix_numpy(xyz):
    """From https://gist.github.com/azag0/c40237b4a746a01b9a79"""
    return np.sqrt(np.sum((xyz[None, :]-xyz[:, None])**2, 2))


def dist_vec_naive(xyz, L):
    N = xyz.shape[0]
    cell = L*np.eye(3)
    inv_cell = np.linalg.inv(cell)
    dist_vec = np.zeros(N*(N-1)//2)
    cnt = 0
    for i in range(N):
        for j in range(i):
            dr = xyz[i] - xyz[j]
            G = np.dot(inv_cell, dr)
            G_n = G - np.round(G)
            dr_n = np.dot(cell, G_n)
            dist_vec[cnt] = np.linalg.norm(dr_n)
            cnt += 1
    return dist_vec


if __name__ == "__main__":
    args = docopt(__doc__)
    L = float(args["--L"])
    Nb = int(args["--bins"])
    infiles = glob.glob(args["<infiles>"])
    print(infiles)

    bins = np.linspace(0.0, L, Nb+1)
    r = bins[:-1] + np.diff(bins)[0]/2.0
    dr = r[1] - r[0]
    rdf = np.zeros((Nb, 2))
    rdf[:, 0] = r

    for infile in infiles:
        # A = np.loadtxt(infile)
        A = read_xyzmatrix(infile)[:, 1:]
        dist_vec = dist_vec_naive(A, L)
        h, _ = np.histogram(dist_vec, bins=bins)
        rdf[:, 1] += h

    rdf[:, 1] *= 1./float(len(infiles)) * L**3/len(dist_vec) / (4*np.pi*r**2 * dr)
    np.savetxt("rdf.out", rdf)
    print("rdf saved into rdf.out")
