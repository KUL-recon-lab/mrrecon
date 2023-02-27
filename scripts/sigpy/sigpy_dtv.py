"""minimal script that shows how to solve L2square data fidelity + anatomical DTV prior"""
import sigpy
import cupy as cp
import numpy as np

import pymirc.viewer as pv

ishape = (64, 64, 64)
num_k_space_points = np.prod(ishape) // 4
noise_level = 3e-2
beta = 1e0

#--------------------------------------------------------------------------

x = cp.zeros(ishape, dtype=cp.complex64)
x[((1 * ishape[0]) // 4):((3 * ishape[0]) // 4),
  ((1 * ishape[1]) // 4):((3 * ishape[1]) // 4),
  ((1 * ishape[2]) // 4):((3 * ishape[2]) // 4), ] = 1

x[((3 * ishape[0]) // 8):((5 * ishape[0]) // 8),
  ((3 * ishape[1]) // 8):((5 * ishape[1]) // 8),
  ((3 * ishape[2]) // 8):((5 * ishape[2]) // 8), ] = 2

# setup a (composite) forward operator
coords = (ishape[0] / 2) * (2 * cp.random.rand(num_k_space_points, 3) - 1)
A = sigpy.linop.NUFFT(ishape, coords, oversamp=1.25, width=4)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# setup projected gradient operator for DTV

# set up the operator for regularization
G = sigpy.linop.FiniteDifference(ishape, axes=None)

# setup a joint gradient field
prior_image = -34.1 * x
xi = G(prior_image)

# normalize the real and imaginary part of the joint gradient field
real_norm = cp.linalg.norm(xi.real, axis=0)
imag_norm = cp.linalg.norm(xi.imag, axis=0)

ir = cp.where(real_norm > 0)
ii = cp.where(imag_norm > 0)

for i in range(xi.shape[0]):
    xi[i, ...].real[ir] /= real_norm[ir]
    xi[i, ...].imag[ii] /= imag_norm[ii]

M = sigpy.linop.Multiply(G.oshape, xi)
S = sigpy.linop.Sum(M.oshape, (0, ))
I = sigpy.linop.Identity(M.oshape)

# projection operator
P = I - (M.H * S.H * S * M)

# projected gradient operator
PG = P * G

#--------------------------------------------------------------------------

# simulate noise-free data
y = A.apply(x)

# add noise to the data
y += noise_level * cp.abs(
    y.max()) * (cp.random.randn(*y.shape) + 1j * cp.random.randn(*y.shape))

proxg = sigpy.prox.L1Reg(G.oshape, lamda=beta)  # define proximal operator

alg = sigpy.app.LinearLeastSquares(A, y, G=PG, proxg=proxg, max_iter=500)
#alg = sigpy.app.LinearLeastSquares(A, y, G=G, proxg=proxg, max_iter=500)

x_hat = alg.run()

x_hat_cpu = cp.asnumpy(x_hat)
vi = pv.ThreeAxisViewer([np.abs(x_hat_cpu), x_hat_cpu.real, x_hat_cpu.imag])