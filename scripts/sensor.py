# -*- coding: utf-8 -*-

from . import Graph
from pygsp.utils import distanz
from pygsp.graphs.gutils import check_connectivity

import numpy as np
from scipy import sparse
from math import ceil, sqrt, log


class Sensor(Graph):
    r"""
    Create a random sensor graph.

    Parameters
    ----------
    N : int
        Number of nodes (default = 64)
    Nc : int
        Minimum number of connections (default = 1)
    regular : bool
        Flag to fix the number of connections to nc (default = False)
    n_try : int
        Number of attempt to create the graph (default = 50)
    distribute : bool
        To distribute the points more evenly (default = False)
    connected : bool
        To force the graph to be connected (default = True)

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Sensor(N=300)

    """
    def get_nc_connection(self, W, param_nc):
        Wtmp = W
        W = np.zeros(np.shape(W))

        for i in range(np.shape(W)[0]):
            l = Wtmp[i]
            for j in range(param_nc):
                val = np.max(l)
                ind = np.argmax(l)
                W[i, ind] = val
                l[ind] = 0

        W = (W + np.conjugate(W).T)/2.

        return W

    def create_weight_matrix(self, N, param_distribute, param_regular, param_Nc):
        XCoords = np.zeros((N, 1))
        YCoords = np.zeros((N, 1))

        if param_distribute:
            mdim = int(ceil(sqrt(N)))
            for i in range(mdim):
                for j in range(mdim):
                    if i*mdim + j < N:
                        XCoords[i*mdim + j] = \
                            np.array(1./float(mdim)*np.random.rand()
                                     + i/float(mdim))
                        YCoords[i*mdim + j] = \
                            np.array(1./float(mdim)*np.random.rand()
                                     + j/float(mdim))

        # take random coordinates in a 1 by 1 square
        else:
            XCoords = np.random.rand(N, 1)
            YCoords = np.random.rand(N, 1)

        Coords = np.concatenate((XCoords, YCoords), axis=1)

        # Compute the distanz between all the points
        target_dist_cutoff = 2*N**(-0.5)
        T = 0.6
        s = sqrt(-target_dist_cutoff**2/(2*log(T)))
        d = distanz(x=Coords.T)
        W = np.exp(-d**2/(2.*s**2))
        W -= np.diag(np.diag(W))

        if param_regular:
            W = self.get_nc_connection(W, param_Nc)

        else:
            W2 = self.get_nc_connection(W, param_Nc)
            W = np.where(W < T, 0, W)
            W = np.where(W2 > 0, W2, W)

        return W, Coords

    def __init__(self, N=64, Nc=2, regular=False, n_try=50,
                 distribute=False, connected=True, **kwargs):

        self.N = N
        self.Nc = Nc
        self.regular = regular
        self.n_try = n_try
        self.distribute = distribute
        self.connected = connected

        if self.connected:
            for x in range(self.n_try):
                W, Coords = self.create_weight_matrix(self.N,
                                                      self.distribute,
                                                      self.regular,
                                                      self.Nc)

                self.W = W
                self.A = sparse.lil_matrix(self.W > 0)

                if check_connectivity(self):
                    break

                elif x == self.n_try - 1:
                    self.logger.warning("Graph is not connected")

        else:
            W, Coords = self.create_weight_matrix(self.N, self.distribute,
                                                  self.regular, self.Nc)

        W = sparse.lil_matrix(W)
        self.W = (W + W.getH())/2.
        self.coords = Coords

        if self.regular:
            self.gtype = "regular sensor"
        else:
            self.gtype = "sensor"
        self.directed = False

        self.plotting = {"limits": np.array([0, 1, 0, 1])}

        super(Sensor, self).__init__(W=self.W, N=self.N, coords=self.coords,
                                     plotting=self.plotting, gtype=self.gtype,
                                     directed=self.directed, **kwargs)
