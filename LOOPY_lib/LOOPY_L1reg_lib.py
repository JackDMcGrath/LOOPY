#!/usr/bin/env python3
"""
# L1-norm regularized least-squares solver, or LASSO (least absolute shrinkage and selection operator)
#
# Modified from CVXOPT-1.2 on GitHub (examples/doc/chap8/l1regls.py)
# by Zhang Yunjun, 10 Jan 2019 (add alpha argument)
#
# Another implementation is in cvxpy on GitHub (examples/lasso.py),
# which also support integer solution. But this implementation is very slow.
#
# Reference:
# Andersen, M., J. Dahl, Z. Liu, and L. Vandenberghe (2011), Interior-point methods for large-scale
# cone programming, in Optimization for machine learning, edited by S. Sra, S. Nowozin and S. J. Wright,
# MIT Press.
# Yunjun, Z., Fattahi, H., Amelung, F. (2019), Small baseline InSAR time series analysis: Unwrapping error
# correction and noise reduction, Computers & Geosciences, 133

"""
# %% Change log
'''
v1.0.0 20230321 Jack McGrath, Uni of Leeds
 - Initial implementation based of LiCSBAS13_invert_small_baselines.py
'''


import numpy as np
import math

from cvxopt import (
    blas,
    div,
    lapack,
    matrix,
    mul,
    solvers,
    spdiag,
    sqrt,
)


def l1regls(A, y, alpha=1.0, show_progress=1):
    """
    Returns the solution of l1-norm regularized least-squares problem
        minimize || A*x - y ||_2^2  + alpha * || x ||_1.
    Parameters: A : 2D cvxopt.matrix for the design matrix in (m, n)
                y : 2D cvxopt.matrix for the observation in (m, 1)
                alpha : float for the degree of shrinkage
                show_progress : bool, show solving progress
    Returns:    x : 2D cvxopt.matrix in (m, 1)
    Example:    A = matrix(np.array(-C, dtype=float))
                b = matrix(np.array(closure_int, dtype=float).reshape(C.shape[0], -1))
                x = np.round(l1reg_lstsq(A, b, alpha=1e-2))
    """
    solvers.options['show_progress'] = show_progress

    m, n = A.size
    q = matrix(alpha, (2 * n, 1))
    q[:n] = -2.0 * A.T * y

    def P(u, v, alpha=1.0, beta=0.0):
        """
            v := alpha * 2.0 * [ A'*A, 0; 0, 0 ] * u + beta * v
        """
        v *= beta
        v[:n] += alpha * 2.0 * A.T * (A * u[:n])

    def G(u, v, alpha=1.0, beta=0.0, trans='N'):
        """
            v := alpha*[I, -I; -I, -I] * u + beta * v  (trans = 'N' or 'T')
        """

        v *= beta
        v[:n] += alpha * (u[:n] - u[n:])
        v[n:] += alpha * (-u[:n] - u[n:])

    h = matrix(0.0, (2 * n, 1))

    # Customized solver for the KKT system
    #
    #     [  2.0*A'*A  0    I      -I     ] [x[:n] ]     [bx[:n] ]
    #     [  0         0   -I      -I     ] [x[n:] ]  =  [bx[n:] ].
    #     [  I        -I   -D1^-1   0     ] [zl[:n]]     [bzl[:n]]
    #     [ -I        -I    0      -D2^-1 ] [zl[n:]]     [bzl[n:]]
    #
    # where D1 = W['di'][:n]**2, D2 = W['di'][n:]**2.
    #
    # We first eliminate zl and x[n:]:
    #
    #     ( 2*A'*A + 4*D1*D2*(D1+D2)^-1 ) * x[:n] =
    #         bx[:n] - (D2-D1)*(D1+D2)^-1 * bx[n:] +
    #         D1 * ( I + (D2-D1)*(D1+D2)^-1 ) * bzl[:n] -
    #         D2 * ( I - (D2-D1)*(D1+D2)^-1 ) * bzl[n:]
    #
    #     x[n:] = (D1+D2)^-1 * ( bx[n:] - D1*bzl[:n]  - D2*bzl[n:] )
    #         - (D2-D1)*(D1+D2)^-1 * x[:n]
    #
    #     zl[:n] = D1 * ( x[:n] - x[n:] - bzl[:n] )
    #     zl[n:] = D2 * (-x[:n] - x[n:] - bzl[n:] ).
    #
    # The first equation has the form
    #
    #     (A'*A + D)*x[:n]  =  rhs
    #
    # and is equivalent to
    #
    #     [ D    A' ] [ x:n] ]  = [ rhs ]
    #     [ A   -I  ] [ v    ]    [ 0   ].
    #
    # It can be solved as
    #
    #     ( A*D^-1*A' + I ) * v = A * D^-1 * rhs
    #     x[:n] = D^-1 * ( rhs - A'*v ).

    S = matrix(0.0, (m, m))
    Asc = matrix(0.0, (m, n))
    v = matrix(0.0, (m, 1))

    def Fkkt(W):

        # Factor
        #
        #     S = A*D^-1*A' + I
        #
        # where D = 2*D1*D2*(D1+D2)^-1, D1 = d[:n]**-2, D2 = d[n:]**-2.

        d1, d2 = W['di'][:n]**2, W['di'][n:]**2

        # ds is square root of diagonal of D
        ds = math.sqrt(2.0) * div(mul(W['di'][:n], W['di'][n:]),
                                  sqrt(d1 + d2))
        d3 = div(d2 - d1, d1 + d2)

        # Asc = A*diag(d)^-1/2
        Asc = A * spdiag(ds**-1)

        # S = I + A * D^-1 * A'
        blas.syrk(Asc, S)
        S[::m + 1] += 1.0
        lapack.potrf(S)

        def g(x, y, z):

            x[:n] = 0.5 * (x[:n] - mul(d3, x[n:]) +
                           mul(d1, z[:n] + mul(d3, z[:n])) - mul(d2, z[n:] -
                                                                   mul(d3, z[n:])) )
            x[:n] = div(x[:n], ds)

            # Solve
            #
            #     S * v = 0.5 * A * D^-1 * ( bx[:n] -
            #         (D2-D1)*(D1+D2)^-1 * bx[n:] +
            #         D1 * ( I + (D2-D1)*(D1+D2)^-1 ) * bzl[:n] -
            #         D2 * ( I - (D2-D1)*(D1+D2)^-1 ) * bzl[n:] )

            blas.gemv(Asc, x, v)
            lapack.potrs(S, v)

            # x[:n] = D^-1 * ( rhs - A'*v ).
            blas.gemv(Asc, v, x, alpha=-1.0, beta=1.0, trans='T')
            x[:n] = div(x[:n], ds)

            # x[n:] = (D1+D2)^-1 * ( bx[n:] - D1*bzl[:n]  - D2*bzl[n:] )
            #         - (D2-D1)*(D1+D2)^-1 * x[:n]
            x[n:] = div(x[n:] - mul(d1, z[:n]) - mul(d2, z[n:]), d1 + d2)\
                - mul(d3, x[:n])

            # zl[:n] = D1^1/2 * (  x[:n] - x[n:] - bzl[:n] )
            # zl[n:] = D2^1/2 * ( -x[:n] - x[n:] - bzl[n:] ).
            z[:n] = mul(W['di'][:n], x[:n] - x[n:] - z[:n])
            z[n:] = mul(W['di'][n:], -x[:n] - x[n:] - z[n:])

        return g

    return solvers.coneqp(P, q, G, h, kktsolver=Fkkt)['x'][:n]



def linsolve(G, d):
    """
    Simple BLUE estimator
    """
    m = np.matmul(np.matmul(np.linalg.matrix_power(np.matmul(G.T, G), -1), G.T), d)
    return m


def makeLoopClosure(nIfg, nepoch, epoch_ix, ifg_ix, ifg_disp, wrap):
    """
    Calculate the loops and loop closure values for a pixel in of all IFGs in SB network 

    Parameters
    ----------
    nIfg : Float
        Number of IFGs.
    nepoch : float
        Number of Epochs.
    epoch_ix : nepoch x 1 Array
        Array of epoch indicies (where epoch 1 is 1)
    ifg_ix : nIFg x 2 array
        Array containing the epoch indicies of each IFG
    ifg_disp : nIFG x 1 array
        Displacement of each IFG 
    wrap : TYPE
        DESCRIPTION.

    Returns
    -------
    loop : TYPE
        DESCRIPTION.
    closure : TYPE
        DESCRIPTION.

    """
    loop = np.zeros((1, nIfg))
    closure = np.array([0])
    # modDisp = (ifg_disp / wrap).round()  # Modulo 2 pi displacements
    modDisp = ifg_disp.copy()

    for ii in range(0, nepoch):
        e1 = epoch_ix[ii] + 1
        ifg1 = np.where(ifg_ix[:, 0] == e1)[0]
        for jj in range(0, ifg1.shape[0]):
            e2 = ifg_ix[ifg1[jj], 1]
            ifg2 = np.where(ifg_ix[:, 0] == e2)[0]
            for kk in range(0, ifg2.shape[0]):
                e3 = ifg_ix[ifg2[kk], 1]
                if np.where((ifg_ix == (e1, e3)).all(axis=1))[0].size:
                    newloop = np.zeros((1, nIfg))
                    ifg12 = np.where((ifg_ix == (e1, e2)).all(axis=1))[0][0]
                    ifg13 = np.where((ifg_ix == (e1, e3)).all(axis=1))[0][0]
                    ifg23 = np.where((ifg_ix == (e2, e3)).all(axis=1))[0][0]
                    newloop[0, np.array([ifg12, ifg23, ifg13])] = [1, 1, -1]
                    loop = np.vstack([loop, newloop])
                    closure = np.vstack([closure, modDisp[ifg12] + modDisp[ifg23] - modDisp[ifg13]])

    loop = np.delete(loop, 0, axis=0)
    closure = np.delete(closure, 0, axis=0)
    closure = (closure / wrap).round()

    return loop, closure
