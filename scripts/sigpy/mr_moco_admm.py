"""minimal example of gated MR recon with known motion fields and regularization on image

   learnings:
      - ADMM converges for wide range of rho (e.g. 1e-2 ... 1e2)
      - since u is the scaled dual (y/rho), it looks very different at convergence
      -> for motion estimation, where we match S(lam) to (z + u), having a small rho seems bad
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

import sigpy
from copy import deepcopy

from utils_moco import golden_angle_2d_readout, stacked_nufft_operator

# input parameters

# number of gates
num_gates = 6

# total number of k-space spokes distributed over the gates
num_spokes = 50
# number of k-space points per spoke
num_points = 128

# noise level of data
noise_level = 1e-2

# rho parameter of ADMM
rho = 1e-1
# weight of TV prior for images
beta = 1e-2
# number of ADMM iterations
num_iter = 30

# number of PDHG iterations for ADMM subproblem (2)
max_num_iter_subproblem_2 = 100
sigma_pdhg = 1.

# bool whether to solve the original or approximated subproblem (2)
use_subproblem2_approx = True

# random seed
seed = 1
np.random.seed(seed)

#--------------------------------------------------------------------

# setup the ground truth image - "fake" 3D image from 2D slice
gt = cp.load('3d_test_mri.npz')['image']

gt /= gt.max()
gt = gt - 0.5 * 1j * gt
gt = gt.astype(cp.complex64)

sim_img_shape = gt.shape

# setup the image shape for reconstruction
# for convenience, we use exactly half of the simulation shape
# to be able to downscale the circ shift operators
img_shape = tuple([x // 2 for x in sim_img_shape])

# setup all k-space trajectories
all_ks = golden_angle_2d_readout(img_shape[0] // 2, num_spokes, num_points)

# distribute the k-space trajectories to the different gates
tmp = np.arange(num_spokes)
np.random.shuffle(tmp)

# a list for the kspace coordinates per gates (can have different size per gate)
k_gate = []
# list of corresponding Fourier operator per gate for simulation
Fs_sim = []
# list of corresponding Fourier operator per gate for reconstruction
Fs = []
# list of motion warping operators acting on highres sim grid
Ss_true = []
# list of motion warping operators acting on lowres recon grid
Ss_true_downsampled = []
# list of data
ds_noise_free = []
ds = []

for i in range(num_gates):
    k_gate.append(all_ks[i::num_gates, ...])

    # setup the Fourier operators for the data simulation (acting on a finer grid)
    Fs_sim.append(stacked_nufft_operator(sim_img_shape, k_gate[i]))

    if i == 0:
        # we treat the first gate as reference (no displacement)
        Ss_true.append(sigpy.linop.Identity(sim_img_shape))
        Ss_true_downsampled.append(sigpy.linop.Identity(img_shape))
    else:
        shift = 2 * int((sim_img_shape[-1] // 16) * np.random.randn(1))
        if i == 1:
            ax = 1
        elif i == 2:
            ax = 2
        else:
            ax = np.random.randint(len(sim_img_shape) - 1) + 1
        Ss_true.append(
            sigpy.linop.Circshift(sim_img_shape, (shift, ), axes=(ax, )))
        Ss_true_downsampled.append(
            sigpy.linop.Circshift(img_shape, (shift // 2, ), axes=(ax, )))

# normalize the Fourier operators such that |F_k| = 1
max_eig_Fsim = sigpy.app.MaxEig(Fs_sim[0].H * Fs_sim[0],
                                dtype=cp.complex64,
                                max_iter=30).run()

for i in range(num_gates):
    Fs_sim[i] = (1 / np.sqrt(max_eig_Fsim)) * Fs_sim[i]

# simulate the data
for i in range(num_gates):
    ds_noise_free.append((Fs_sim[i] * Ss_true[i])(gt))
    ds.append(ds_noise_free[i] + noise_level *
              (cp.random.randn(*ds_noise_free[i].shape) +
               1j * cp.random.randn(*ds_noise_free[i].shape)))

# crop the data around the center along the FFT axis
start = sim_img_shape[0] // 2 - img_shape[0] // 2
end = start + img_shape[0]

for i in range(num_gates):
    ds_noise_free[i] = ds_noise_free[i][start:end, ...]
    ds[i] = ds[i][start:end, ...]

for i in range(num_gates):
    # setup the Fourier operators for the reconstruction (acting on a coarser grid)
    Fs.append(stacked_nufft_operator(img_shape, k_gate[i]))

# normalize the Fourier operators such that |F_k| = 1
max_eig_F = sigpy.app.MaxEig(Fs[0].H * Fs[0], dtype=cp.complex64,
                             max_iter=30).run()

for i in range(num_gates):
    Fs[i] = (1 / np.sqrt(max_eig_F)) * Fs[i]

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# independent recons to init. z's and estimate intial motion fields
#--------------------------------------------------------------------
#--------------------------------------------------------------------

# gradient operator, factor in front makes sure that |G| = 1
G = sigpy.linop.Gradient(img_shape)

# normalize the norm of the gradient operator
max_eig_G = sigpy.app.MaxEig(G.H * G, dtype=cp.complex64, max_iter=30).run()
G = (1 / np.sqrt(max_eig_G)) * G

# prox for TV prior
proxg_ind = sigpy.prox.L1Reg(G.oshape, beta / num_gates)

ind_recons = cp.zeros((num_gates, *img_shape), dtype=cp.complex64)

for i in range(num_gates):
    alg01 = sigpy.app.LinearLeastSquares(Fs[i],
                                         ds[i],
                                         G=G,
                                         proxg=proxg_ind,
                                         max_iter=100,
                                         sigma=sigma_pdhg)
    ind_recons[i, ...] = alg01.run()

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# initial estimate of motion fields (operators)
#--------------------------------------------------------------------
#--------------------------------------------------------------------

# skipped for now - simply copy the true motion warping operators
Ss = deepcopy(Ss_true_downsampled)

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# reconstruction of all the data without motion modeling as reference
#--------------------------------------------------------------------
#--------------------------------------------------------------------

proxg_sum = sigpy.prox.L1Reg(G.oshape, beta)

alg0 = sigpy.app.LinearLeastSquares(sigpy.linop.Vstack(Fs),
                                    cp.concatenate([x.ravel() for x in ds]),
                                    G=G,
                                    proxg=proxg_sum,
                                    max_iter=500,
                                    sigma=sigma_pdhg)
recon_wo_moco = alg0.run()

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# ADMM
#--------------------------------------------------------------------
#--------------------------------------------------------------------

# prox for subproblem 2 - note extra (1/rho) which is needed for subproblem 2
proxg2 = sigpy.prox.L1Reg(G.oshape, beta / rho)
# prox for subproblem 2 - note extra (1/rho) which is needed for the approximate subproblem 2
proxg2a = sigpy.prox.L1Reg(G.oshape, beta / (num_gates * rho))

# initialize all variables
lam = ind_recons[0, ...].copy()
zs = ind_recons.copy()
us = cp.zeros_like(zs)

cost = np.zeros(num_iter)

recons = cp.zeros((num_iter, *img_shape), dtype=cp.complex64)

for i_outer in range(num_iter):
    ###################################################################
    # subproblem (1) - data fidelity + quadratic - update for z1 and z2
    ###################################################################

    for i in range(num_gates):
        alg11 = sigpy.app.LinearLeastSquares(Fs[i],
                                             ds[i],
                                             x=zs[i],
                                             lamda=rho,
                                             z=(Ss[i](lam) - us[i, ...]))
        zs[i, ...] = alg11.run()

    ###################################################################
    # subproblem (2) - optimize lambda
    ###################################################################

    if use_subproblem2_approx:
        # optimize approximation of subproblem (2) using the inverse
        # of the motion deformation operators
        v = cp.zeros_like(lam)
        for i in range(num_gates):
            ############################################################
            # invert deformation operator here
            ############################################################

            # for "simple" circular shift, the inverse equal the adjoint
            # !!! not true for a non-rigid deformation operator !!!
            S_inv = Ss[i].H
            v += S_inv(us[i] + zs[i])

        v /= num_gates

        if i_outer == 0:
            pdhg_u2a = cp.zeros(G.oshape, dtype=lam.dtype)

        alg2a = sigpy.alg.PrimalDualHybridGradient(
            proxfc=sigpy.prox.Conj(proxg2a),
            proxg=sigpy.prox.L2Reg(img_shape, 1, y=v),
            A=G,
            AH=G.H,
            x=deepcopy(lam),
            u=pdhg_u2a,
            tau=1 / sigma_pdhg,
            sigma=sigma_pdhg)

        for _ in range(max_num_iter_subproblem_2):
            alg2a.update()

        lam = alg2a.x
    else:
        # optimize the exact subproblem (2) which requires knowledge of the
        # adjoint of the motion deformation operators

        S = sigpy.linop.Vstack(Ss)
        y = (us + zs).ravel()

        # we could call LinearLeastSquares directly, but we will use call the
        # PHDG directly which allows us to store the dual variable of PDHG
        # for warm start of the following iteration

        # run PDHG to solve subproblem (2)
        A = sigpy.linop.Vstack([S, G])
        proxfc = sigpy.prox.Stack(
            [sigpy.prox.L2Reg(y.shape, 1, y=-y),
             sigpy.prox.Conj(proxg2)])

        if i_outer == 0:
            max_eig = sigpy.app.MaxEig(A.H * A, dtype=y.dtype,
                                       max_iter=30).run()
            pdhg_u = cp.zeros(A.oshape, dtype=y.dtype)

        alg2 = sigpy.alg.PrimalDualHybridGradient(
            proxfc=proxfc,
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

    # save recon after each outer ADMM iteration
    recons[i_outer, ...] = lam

    ###################################################################
    # update of displacement fields (motion operators) based on z1, z2
    ###################################################################

    # skipped for now

    ###################################################################
    # update of dual variables
    ###################################################################

    # update of dual variables
    for i in range(num_gates):
        us[i] = us[i] + zs[i] - Ss[i](lam)

    ###################################################################
    # evaluation of cost function
    ###################################################################

    # evaluate the cost function
    prior = float(G(lam).real.sum() + G(lam).imag.sum())

    data_fidelity = np.zeros(num_gates)

    for i in range(num_gates):
        e = Fs[i](Ss[i](lam)) - ds[i]
        data_fidelity[i] = float(0.5 * (e.conj() * e).sum().real)

    cost[i_outer] = data_fidelity.sum() + beta * prior

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# visualization
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

ims = dict(vmin=-0.5, vmax=0.5, cmap='gray')
ims2 = dict(vmin=0, vmax=0.7, cmap='gray')

# normalization factor that is due to the different grid sizes in simulation and recon
# (sqrt(2)**3, for a factor a 2 downsampling in all 3 directions)
norm = np.sqrt(8)

# slice to show
sl = img_shape[0] // 2

fig, ax = plt.subplots(5, num_gates, figsize=(num_gates * 2, 5 * 2))
for i in range(num_gates):
    ax[0, i].imshow(cp.asnumpy(zs[i, sl, ...].real) / norm, **ims)
    ax[0, i].set_title(f'z{i} real')
    ax[1, i].imshow(cp.asnumpy(zs[i, sl, ...].imag) / norm, **ims)
    ax[1, i].set_title(f'z{i} imag')
    ax[2, i].imshow(cp.asnumpy(cp.abs(zs[i, sl, ...])) / norm, **ims2)
    ax[2, i].set_title(f'z{i} abs')
    ax[3, i].imshow(cp.asnumpy(cp.abs(Ss[i](lam))[sl, ...]) / norm, **ims2)
    ax[3, i].set_title(f'S{i} lam abs')
    ax[4, i].imshow(cp.asnumpy(cp.abs(Ss_true[i](gt))[sl * 2, ...]), **ims2)
    ax[4, i].set_title(f'S{i} gt abs')
for axx in ax.ravel():
    axx.set_axis_off()
fig.tight_layout()
fig.show()

fig2, ax2 = plt.subplots(4, num_gates, figsize=(num_gates * 2, 4 * 2))
for i in range(num_gates):
    ax2[0, i].imshow(cp.asnumpy(ind_recons[i, sl, ...].real) / norm, **ims)
    ax2[0, i].set_title(f'ind recon{i} real')
    ax2[1, i].imshow(cp.asnumpy(ind_recons[i, sl, ...].imag) / norm, **ims)
    ax2[1, i].set_title(f'ind recon{i} imag')
    ax2[2, i].imshow(cp.asnumpy(cp.abs(ind_recons[i, sl, ...])) / norm, **ims2)
    ax2[2, i].set_title(f'ind recon{i} abs')
    ax2[3, i].imshow(cp.asnumpy(cp.abs(Ss_true[i](gt))[sl * 2, ...]), **ims2)
    ax2[3, i].set_title(f'S{i} gt abs')
for axx in ax2.ravel():
    axx.set_axis_off()
fig2.tight_layout()
fig2.show()

fig3, ax3 = plt.subplots(3, num_gates, figsize=(num_gates * 2, 3 * 2))
for i in range(num_gates):
    ax3[0, i].imshow(
        cp.asnumpy(zs[i, sl, ...].real + us[i, sl, ...].real) / norm, **ims)
    ax3[0, i].set_title(f'z+u{i} real')
    ax3[1, i].imshow(
        cp.asnumpy(zs[i, sl, ...].imag + us[i, sl, ...].imag) / norm, **ims)
    ax3[1, i].set_title(f'z+u{i} imag')
    ax3[2, i].imshow(
        cp.asnumpy(cp.abs(zs[i, sl, ...] + us[i, sl, ...])) / norm, **ims2)
    ax3[2, i].set_title(f'z+u{i} abs')
for axx in ax3.ravel():
    axx.set_axis_off()
fig3.tight_layout()
fig3.show()

fig4, ax4 = plt.subplots(3, 3, figsize=(3 * 2, 3 * 2))
ax4[0, 0].imshow(cp.asnumpy(gt[sl * 2, ...].real), **ims)
ax4[0, 0].set_title(f'gt real')
ax4[1, 0].imshow(cp.asnumpy(gt[sl * 2, ...].imag), **ims)
ax4[1, 0].set_title(f'gt imag')
ax4[2, 0].imshow(cp.asnumpy(cp.abs(gt[sl * 2, ...])), **ims2)
ax4[2, 0].set_title(f'gt abs')
ax4[0, 1].imshow(cp.asnumpy(recon_wo_moco[sl, ...].real) / norm, **ims)
ax4[0, 1].set_title(f'wo moco real')
ax4[1, 1].imshow(cp.asnumpy(recon_wo_moco[sl, ...].imag) / norm, **ims)
ax4[1, 1].set_title(f'wo moco imag')
ax4[2, 1].imshow(cp.asnumpy(cp.abs(recon_wo_moco[sl, ...])) / norm, **ims2)
ax4[2, 1].set_title(f'wo moco abs')
ax4[0, 2].imshow(cp.asnumpy(lam[sl, ...].real) / norm, **ims)
ax4[0, 2].set_title(f'moco real')
ax4[1, 2].imshow(cp.asnumpy(lam[sl, ...].imag) / norm, **ims)
ax4[1, 2].set_title(f'moco imag')
ax4[2, 2].imshow(cp.asnumpy(cp.abs(lam[sl, ...])) / norm, **ims2)
ax4[2, 2].set_title(f'moco abs')
for axx in ax4.ravel():
    axx.set_axis_off()
fig4.tight_layout()
fig4.show()

fig5, ax5 = plt.subplots()
ax5.plot(np.arange(num_iter) + 1, cost)
ax5.set_xlabel('outer iteration')
ax5.set_ylabel('cost')
fig5.tight_layout()
fig5.show()