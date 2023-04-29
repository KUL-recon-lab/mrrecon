"""minimal example of gated MR recon with known motion fields and regularization on image

   learnings:
      - ADMM converges for wide range of rho (e.g. 1e-2 ... 1e2)
      - since u is the scaled dual (y/rho), it looks very different at convergence
      -> for motion estimation, where we match S(lam) to (z + u), having a small rho seems bad
"""

import numpy as np
import cupy as cp
import cupy.typing as cpt
import matplotlib.pyplot as plt

import sigpy
from copy import deepcopy


def golden_angle_2d_readout(kmax: float, num_spokes: int,
                            num_points: int) -> cpt.NDArray:
    """2D golden angle kspace trajectory"""
    tmp = cp.linspace(-kmax, kmax, num_points)
    k = cp.zeros((num_spokes, num_points, 2))

    ga = cp.pi / ((1 + cp.sqrt(5)) / 2)

    for i in range(num_spokes):
        phi = (i * ga) % (2 * cp.pi)
        k[i, :, 0] = tmp * cp.cos(phi)
        k[i, :, 1] = tmp * cp.sin(phi)

    return k


#-------------------------------------------------------------------

# input parameters

# image size
n = 256

# total number of k-space spokes distributed over 3 gates
num_spokes = n // 4
# number of k-space points per spoke
num_points = 4 * n

# noise level of data
noise_level = 1e-2

# rho parameter of ADMM
rho = 1e-1
# weight of TV prior for images
beta = 1e-3
# number of ADMM iterations
num_iter = 100

# number of PDHG iterations for ADMM subproblem (2)
max_num_iter_subproblem_2 = 100
sigma_pdhg = 1.

#--------------------------------------------------------------------

img_shape = (n, n)

# setup the "motion operators"
S1 = sigpy.linop.Identity(
    img_shape)  # we treat the first gate as reference (no displacement)
S2 = sigpy.linop.Circshift(img_shape, (-n // 8, ), axes=(1, ))
S3 = sigpy.linop.Circshift(img_shape, (n // 8, ), axes=(0, ))

# setup all k-space trajectories
all_ks = golden_angle_2d_readout(n // 2, num_spokes, num_points)

# distribute the k-space trajectories to the different gates
k_gate1 = all_ks[0::3, ...]
k_gate2 = all_ks[1::3, ...]
k_gate3 = all_ks[2::3, ...]

# setup the Fourier operators
F1 = sigpy.linop.NUFFT(img_shape, k_gate1)
F2 = sigpy.linop.NUFFT(img_shape, k_gate2)
F3 = sigpy.linop.NUFFT(img_shape, k_gate3)

# normalize the Fourier operators such that |F_k| = 1
max_eig_F1 = sigpy.app.MaxEig(F1.H * F1, dtype=cp.complex64, max_iter=30).run()

F1 = (1 / np.sqrt(max_eig_F1)) * F1
F2 = (1 / np.sqrt(max_eig_F1)) * F2
F3 = (1 / np.sqrt(max_eig_F1)) * F3

gt = cp.flip(cp.load('2d_test_mri.npy').T, 0)
gt /= gt.max()
gt = gt - 0.5 * 1j * gt
gt = gt.astype(cp.complex64)

# simulate the data
d1 = (F1 * S1)(gt)
d2 = (F2 * S2)(gt)
d3 = (F3 * S3)(gt)

# add noise to the data
d1 += noise_level * cp.random.randn(*d1.shape)
d1 += 1j * noise_level * cp.random.randn(*d1.shape)

d2 += noise_level * cp.random.randn(*d2.shape)
d2 += 1j * noise_level * cp.random.randn(*d2.shape)

d3 += noise_level * cp.random.randn(*d3.shape)
d3 += 1j * noise_level * cp.random.randn(*d3.shape)

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# independent recons to init. z's and estimate intial motion fields
#--------------------------------------------------------------------
#--------------------------------------------------------------------

# gradient operator, factor in front makes sure that |G| = 1
G = (1 / np.sqrt(4 * len(img_shape))) * sigpy.linop.Gradient(img_shape)

# prox for TV prior
proxg_ind = sigpy.prox.L1Reg(G.oshape, beta / 3)

alg01 = sigpy.app.LinearLeastSquares(F1,
                                     d1,
                                     G=G,
                                     proxg=proxg_ind,
                                     max_iter=500,
                                     sigma=sigma_pdhg)
ind_recon1 = alg01.run()

alg02 = sigpy.app.LinearLeastSquares(F2,
                                     d2,
                                     G=G,
                                     proxg=proxg_ind,
                                     max_iter=500,
                                     sigma=sigma_pdhg)
ind_recon2 = alg02.run()

alg03 = sigpy.app.LinearLeastSquares(F3,
                                     d3,
                                     G=G,
                                     proxg=proxg_ind,
                                     max_iter=500,
                                     sigma=sigma_pdhg)
ind_recon3 = alg03.run()

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# initial estimate of motion fields (operators)
#--------------------------------------------------------------------
#--------------------------------------------------------------------

# skipped for now

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# ADMM
#--------------------------------------------------------------------
#--------------------------------------------------------------------

# prox for subproblem 2 - note extra (1/rho) which is needed for subproblem 2
proxg2 = sigpy.prox.L1Reg(G.oshape, beta / rho)

# initialize all variables
lam = ind_recon1.copy()
z1 = ind_recon1.copy()
z2 = ind_recon2.copy()
z3 = ind_recon3.copy()
u1 = cp.zeros_like(gt)
u2 = cp.zeros_like(gt)
u3 = cp.zeros_like(gt)

cost = np.zeros(num_iter)

for i in range(num_iter):
    ###################################################################
    # subproblem (1) - data fidelity + quadratic - update for z1 and z2
    ###################################################################

    alg11 = sigpy.app.LinearLeastSquares(F1,
                                         d1,
                                         x=z1,
                                         lamda=rho,
                                         z=(S1(lam) - u1))
    z1 = alg11.run()

    alg12 = sigpy.app.LinearLeastSquares(F2,
                                         d2,
                                         x=z2,
                                         lamda=rho,
                                         z=(S2(lam) - u2))
    z2 = alg12.run()

    alg13 = sigpy.app.LinearLeastSquares(F3,
                                         d3,
                                         x=z3,
                                         lamda=rho,
                                         z=(S3(lam) - u3))
    z3 = alg13.run()

    ###################################################################
    # subproblem (2) - optimize lamda
    ###################################################################
    S = sigpy.linop.Vstack([S1, S2, S3])
    y = cp.concatenate([
        u1.ravel() + z1.ravel(),
        u2.ravel() + z2.ravel(),
        u3.ravel() + z3.ravel()
    ])

    # we could call LinearLeastSquares directly, but we will use call the
    # PHDG directly which allows us to store the dual variable of PDHG
    # for warm start of the following iteration

    #alg2 = sigpy.app.LinearLeastSquares(S,
    #                                    y,
    #                                    x = lam,
    #                                    G=G,
    #                                    proxg=proxg2,
    #                                    max_iter=500,
    #                                    sigma=0.1)

    # run PDHG to solve subproblem (2)
    A = sigpy.linop.Vstack([S, G])
    proxfc = sigpy.prox.Stack(
        [sigpy.prox.L2Reg(y.shape, 1, y=-y),
         sigpy.prox.Conj(proxg2)])

    if i == 0:
        max_eig = sigpy.app.MaxEig(A.H * A, dtype=y.dtype, max_iter=30).run()
        pdhg_u = cp.zeros(A.oshape, dtype=y.dtype)

    alg2 = sigpy.alg.PrimalDualHybridGradient(proxfc=proxfc,
                                              proxg=sigpy.prox.NoOp(A.ishape),
                                              A=A,
                                              AH=A.H,
                                              x=deepcopy(lam),
                                              u=pdhg_u,
                                              tau=1 / (max_eig * sigma_pdhg),
                                              sigma=sigma_pdhg)

    for _ in range(max_num_iter_subproblem_2):
        alg2.update()

    lam = alg2.x

    ###################################################################
    # update of displacement fields (motion operators) based on z1, z2
    ###################################################################

    # skipped for now

    ###################################################################
    # update of dual variables
    ###################################################################

    # update of dual variables
    u1 = u1 + z1 - S1(lam)
    u2 = u2 + z2 - S2(lam)
    u3 = u3 + z3 - S3(lam)

    ###################################################################
    # evaluation of cost function
    ###################################################################

    # evaluate the cost function
    e1 = F1(S1(lam)) - d1
    fid1 = float(0.5 * (e1.conj() * e1).sum().real)
    e2 = F2(S2(lam)) - d2
    fid2 = float(0.5 * (e2.conj() * e2).sum().real)
    e3 = F3(S3(lam)) - d3
    fid3 = float(0.5 * (e3.conj() * e3).sum().real)

    prior = float(G(lam).real.sum() + G(lam).imag.sum())

    cost[i] = fid1 + fid2 + fid3 + beta * prior

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# visualization
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

ims = dict(vmin=-0.5, vmax=0.5, cmap='gray')

fig, ax = plt.subplots(2, 8, figsize=(8 * 2, 2 * 2), sharex=True, sharey=True)
ax[0, 0].imshow(cp.asnumpy(gt.real), **ims)
ax[0, 0].set_title('gt real')
ax[1, 0].imshow(cp.asnumpy(gt.imag), **ims)
ax[1, 0].set_title('gt imag')
ax[0, 1].imshow(cp.asnumpy(lam.real), **ims)
ax[0, 1].set_title('lambda real')
ax[1, 1].imshow(cp.asnumpy(lam.imag), **ims)
ax[1, 1].set_title('lambda imag')
ax[0, 2].imshow(cp.asnumpy(u1.real), **ims)
ax[0, 2].set_title('u1 real')
ax[1, 2].imshow(cp.asnumpy(u1.imag), **ims)
ax[1, 2].set_title('u1 imag')
ax[0, 3].imshow(cp.asnumpy(u2.real), **ims)
ax[0, 3].set_title('u2 real')
ax[1, 3].imshow(cp.asnumpy(u2.imag), **ims)
ax[1, 3].set_title('u2 imag')
ax[0, 4].imshow(cp.asnumpy(u3.real), **ims)
ax[0, 4].set_title('u3 real')
ax[1, 4].imshow(cp.asnumpy(u3.imag), **ims)
ax[1, 4].set_title('u3 imag')

ax[0, 5].imshow(cp.asnumpy(z1.real), **ims)
ax[0, 5].set_title('z1 real')
ax[1, 5].imshow(cp.asnumpy(z1.imag), **ims)
ax[1, 5].set_title('z1 imag')
ax[0, 6].imshow(cp.asnumpy(z2.real), **ims)
ax[0, 6].set_title('z2 real')
ax[1, 6].imshow(cp.asnumpy(z2.imag), **ims)
ax[1, 6].set_title('z2 imag')
ax[0, 7].imshow(cp.asnumpy(z3.real), **ims)
ax[0, 7].set_title('z3 real')
ax[1, 7].imshow(cp.asnumpy(z3.imag), **ims)
ax[1, 7].set_title('z3 imag')

for axx in ax.ravel():
    axx.set_axis_off()

fig.tight_layout()
fig.show()

fig2, ax2 = plt.subplots()
ax2.plot(cost)
fig2.tight_layout()
fig2.show()

fig3, ax3 = plt.subplots(2,
                         3,
                         figsize=(3 * 2, 2 * 2),
                         sharex=True,
                         sharey=True)
ax3[0, 0].imshow(cp.asnumpy(ind_recon1.real), **ims)
ax3[0, 0].set_title('ind recon 1 real')
ax3[1, 0].imshow(cp.asnumpy(ind_recon1.imag), **ims)
ax3[1, 0].set_title('ind recon 1 imag')
ax3[0, 1].imshow(cp.asnumpy(ind_recon2.real), **ims)
ax3[0, 1].set_title('ind recon 2 real')
ax3[1, 1].imshow(cp.asnumpy(ind_recon2.imag), **ims)
ax3[1, 1].set_title('ind recon 2 imag')
ax3[0, 2].imshow(cp.asnumpy(ind_recon3.real), **ims)
ax3[0, 2].set_title('ind recon 3 real')
ax3[1, 2].imshow(cp.asnumpy(ind_recon3.imag), **ims)
ax3[1, 2].set_title('ind recon 3 imag')

for axx in ax3.ravel():
    axx.set_axis_off()

fig3.tight_layout()
fig3.show()
