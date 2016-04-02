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


def dmatrix_numpy(xyz):
    """From https://gist.github.com/azag0/c40237b4a746a01b9a79"""
    return np.sqrt(np.sum((xyz[None, :]-xyz[:, None])**2, 2))

if __name__ == "__main__":
    args = docopt(__doc__)
    L = float(args["--L"])
    Nb = int(args["--bins"])
    infiles = glob.glob(args["<infiles>"])
    print(infiles)

    bins = np.linspace(0.0, L, Nb+1)
    rdf = np.zeros((Nb-1, 2))
    rdf[:, 0] = bins[:-1] + np.diff(bins)[0]/2.0

    for infile in infiles:
        A = np.loadtxt(infile)
        dist_mat = dmatrix_numpy(A)
        h, bins = np.histogram(A, bins=bins)
        rdf[:, 1] += h

    np.savetxt("rdf.out", rdf)

