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

import os
import math
import shutil
import numpy as np
import LiCSBAS_tools_lib as tools_lib
import matplotlib.pyplot as plt

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

cmap_wrap = tools_lib.get_cmap('SCM.romaO')
cmap_corr = tools_lib.get_cmap('SCM.vik')

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
                           mul(d1, z[:n] + mul(d3, z[:n])) -
                           mul(d2, z[n:] - mul(d3, z[n:])))
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


# %%
def get_patchpix(n_pt_unnan, n_ifg, n_loop, memory_size, n_para, extra=5):
    """
    Get patch number of pixels for memory size (in MB). 1 data point if 4 bytes
    for float32. Take availiable amount of memory, divide between n_para, and
    see how many pixels can be inverted at once, given G = (mxn) * n_pt_unnan and
    d = (m*n_pt_unnan)x1 for all data

    Returns:
        n_patch : Number of patches (int)
        patchrow : List of the number of pixels for each patch
                ex) [[0, 1234], [1235, 2469],... ]
    """
    data_max = np.sqrt((memory_size / 4) * (2 ** 20) / n_ifg / n_loop / n_para / extra)  # from MB to bytes, 4byte floats
    n_patch = int(np.ceil(n_pt_unnan / data_max))  # Number of patches needed

    patchpix = []
    for i in range(n_patch):
        pixspacing = int(np.ceil(n_pt_unnan / n_patch))
        patchpix.append([i * pixspacing, (i + 1) * pixspacing])
        if i == n_patch - 1:
            patchpix[-1][-1] = n_pt_unnan

    return n_patch, patchpix


# %%
def reset_corrections(ifgdir):
    """
    Remove previous corrections
    """
    if os.path.exists(os.path.join(ifgdir, 'L1_correction_png')):
        shutil.rmtree(os.path.join(ifgdir, 'L1_correction_png'))

    for root, dirs, files in os.walk(ifgdir):
        for dir_name in dirs:
            unwfile = os.path.join(root, dir_name, dir_name + '.unw')
            unwpngfile = os.path.join(root, dir_name, dir_name + '.unw.png')
            uncorrfile = os.path.join(root, dir_name, dir_name + '.unw_uncorr')
            uncorrpngfile = os.path.join(root, dir_name, dir_name + '.unw_uncorr.png')
            corrcomppng = os.path.join(root, dir_name, dir_name + '.L1_compare.png')
            corrfile = os.path.join(root, dir_name, dir_name + '.L1_corr')
            if os.path.exists(uncorrfile):
                # Remove correction files
                if os.path.exists(unwfile):
                    os.remove(unwfile)
                if os.path.exists(unwpngfile):
                    os.remove(unwpngfile)
                if os.path.exists(corrcomppng):
                    os.remove(corrcomppng)
                if os.path.exists(corrfile):
                    os.remove(corrfile)
                # Rename uncorrected files
                shutil.move(uncorrfile, unwfile)
            if os.path.exists(uncorrpngfile):
                shutil.move(uncorrpngfile, unwpngfile)


# %%
def make_loop_png(uncorr, corrunw, npi, corr, png, titles4, cycle):

    # Settings
    plt.rcParams['axes.titlesize'] = 10
    ifg = [uncorr, corrunw]

    length, width = uncorr.shape
    if length > width:
        figsize_y = 10
        figsize_x = int((figsize_y - 1) * width / length)
        if figsize_x < 5:
            figsize_x = 5
    else:
        figsize_x = 10
        figsize_y = int(figsize_x * length / width + 1)
        if figsize_y < 3:
            figsize_y = 3

    # Plot
    fig = plt.figure(figsize=(figsize_x, figsize_y))

    # Original and Corrected unw
    for i in range(2):
        data_wrapped = np.angle(np.exp(1j * (ifg[i] / cycle)) * cycle)
        ax = fig.add_subplot(2, 2, i + 1)  # index start from 1
        im = ax.imshow(data_wrapped, vmin=-np.pi, vmax=+np.pi, cmap=cmap_wrap,
                       interpolation='nearest')
        ax.set_title('{}'.format(titles4[i]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        cax = plt.colorbar(im)
        cax.set_ticks([])

    # npi
    ax = fig.add_subplot(2, 2, 3)  # index start from 1
    im = ax.imshow(npi, cmap='tab20c', interpolation='nearest')
    ax.set_title('{}'.format(titles4[2]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cax = plt.colorbar(im)

    # Correction
    ax = fig.add_subplot(2, 2, 4)  # index start from 1
    im = ax.imshow(corr, vmin=-np.nanmax(abs(corr)), vmax=np.nanmax(abs(corr)),
                   cmap=cmap_corr, interpolation='nearest')
    ax.set_title('{}'.format(titles4[3]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cax = plt.colorbar(im)

    plt.tight_layout()
    plt.savefig(png)
    plt.close()


# %%
def linsolve(G, d):
    """
    Simple BLUE estimator (BAD - only using for mapping testing)
    """
    m = np.matmul(np.matmul(np.linalg.matrix_power(np.matmul(G.T, G), -1), G.T), d)
    return m
