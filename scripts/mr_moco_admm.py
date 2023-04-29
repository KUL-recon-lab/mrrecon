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

n0 = 64
n1 = 64
img_shape = (n0, n1)

noise_level = 3e-1

rho = 1e0
beta = 3e0

num_iter = 250

max_num_iter_pdhg = 100
sigma_pdhg = 1.

#--------------------------------------------------------------------

# setup the "motion operators"
S1 = sigpy.linop.Circshift(img_shape, (n0 // 8, ), axes=(0, ))
S2 = sigpy.linop.Circshift(img_shape, (-n0 // 8, ), axes=(1, ))

# setup the Fourier operators
F1 = sigpy.linop.FFT(img_shape, center=True)
F2 = sigpy.linop.FFT(img_shape, center=True)

# setup the ground truth image
gt = cp.zeros(img_shape, dtype=np.complex64)
gt[(1 * n0 // 4):(3 * n0 // 4), (1 * n1 // 4):(3 * n1 // 4)] = 1 - 1j
gt[(3 * n0 // 8):(5 * n0 // 8), (3 * n1 // 5):(3 * n1 // 8)] = -0.5 + 0.5j

# simulate the data
d1 = (F1 * S1)(gt)
d2 = (F2 * S2)(gt)

# add noise to the data
d1 += noise_level * cp.random.randn(*d1.shape)
d1 += 1j * noise_level * cp.random.randn(*d1.shape)

d2 += noise_level * cp.random.randn(*d2.shape)
d2 += 1j * noise_level * cp.random.randn(*d2.shape)

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# ADMM
#--------------------------------------------------------------------
#--------------------------------------------------------------------

# gradient operator
G = (1 / np.sqrt(8)) * sigpy.linop.Gradient(img_shape)

# prox for subproblem 2 - note extra (1/rho) which is needed for subproblem 2
proxg2 = sigpy.prox.L1Reg(G.oshape, beta / rho)

# initialize all variables
lam = cp.zeros_like(gt)
z1 = cp.zeros_like(gt)
z2 = cp.zeros_like(gt)
u1 = cp.zeros_like(gt)
u2 = cp.zeros_like(gt)

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

    ###################################################################
    # subproblem (2) - optimize lamda
    ###################################################################
    S = sigpy.linop.Vstack([S1, S2])
    y = cp.concatenate([u1.ravel() + z1.ravel(), u2.ravel() + z2.ravel()])

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

    for _ in range(max_num_iter_pdhg):
        alg2.update()

    lam = alg2.x

    # update of dual variables
    u1 = u1 + z1 - S1(lam)
    u2 = u2 + z2 - S2(lam)

    # evaluate the cost function
    e1 = F1(S1(lam)) - d1
    fid1 = float(0.5 * (e1.conj() * e1).sum().real)
    e2 = F2(S2(lam)) - d2
    fid2 = float(0.5 * (e2.conj() * e2).sum().real)

    prior = float(G(lam).real.sum() + G(lam).imag.sum())

    cost[i] = fid1 + fid2 + beta * prior

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# visualization
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

ims = dict(vmin=-1, vmax=1, cmap='gray')

fig, ax = plt.subplots(2, 9, figsize=(9 * 2, 2 * 2))
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
ax[0, 4].imshow(cp.asnumpy(z1.real), **ims)
ax[0, 4].set_title('z1 real')
ax[1, 4].imshow(cp.asnumpy(z1.imag), **ims)
ax[1, 4].set_title('z1 imag')
ax[0, 5].imshow(cp.asnumpy(z2.real), **ims)
ax[0, 5].set_title('z2 real')
ax[1, 5].imshow(cp.asnumpy(z2.imag), **ims)
ax[1, 5].set_title('z2 imag')
ax[0, 6].imshow(cp.asnumpy(rho * u1.real), **ims)
ax[0, 6].set_title('rho*u1 real')
ax[1, 6].imshow(cp.asnumpy(rho * u1.imag), **ims)
ax[1, 6].set_title('rho*u1 imag')
ax[0, 7].imshow(cp.asnumpy(u1.real + z1.real), **ims)
ax[0, 7].set_title('u1+z1 real')
ax[1, 7].imshow(cp.asnumpy(u1.imag + z1.real), **ims)
ax[1, 7].set_title('u1+z1 imag')
ax[0, 8].imshow(cp.asnumpy(u2.real + z2.real), **ims)
ax[0, 8].set_title('u2+z2 real')
ax[1, 8].imshow(cp.asnumpy(u2.imag + z2.real), **ims)
ax[1, 8].set_title('u2+z2 imag')

for axx in ax.ravel():
    axx.set_axis_off()

fig.tight_layout()
fig.show()

fig2, ax2 = plt.subplots()
ax2.plot(cost)
fig2.tight_layout()
fig2.show()