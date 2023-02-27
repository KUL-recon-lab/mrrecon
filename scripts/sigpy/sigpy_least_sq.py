"""minimal script that shows how to solve L2square data fidelity + regularization problem using sigpy"""
import sigpy
import cupy as cp
import numpy as np

import pymirc.viewer as pv

ishape = (64, 64, 64)
num_k_space_points = np.prod(ishape)
beta = 1e-1

#--------------------------------------------------------------------------

x = cp.zeros(ishape, dtype=cp.complex64)
x[((1 * ishape[0]) // 4):((3 * ishape[0]) // 4),
  ((1 * ishape[1]) // 4):((3 * ishape[1]) // 4),
  ((1 * ishape[2]) // 4):((3 * ishape[2]) // 4), ] = 1

x[((3 * ishape[0]) // 8):((5 * ishape[0]) // 8),
  ((3 * ishape[1]) // 8):((5 * ishape[1]) // 8),
  ((3 * ishape[2]) // 8):((5 * ishape[2]) // 8), ] = 2

# setup a (composite) forward operator
coords1 = (ishape[0] / 2) * (2 * cp.random.rand(num_k_space_points, 3) - 1)
A1 = sigpy.linop.NUFFT(ishape, coords1, oversamp=1.25, width=4)
b1 = cp.ones(ishape)
B1 = sigpy.linop.Multiply(ishape, b1)
C1 = A1 * B1

coords2 = (ishape[0] / 2) * (2 * cp.random.rand(num_k_space_points, 3) - 1)
A2 = sigpy.linop.NUFFT(ishape, coords2, oversamp=1.25, width=4)
b2 = cp.ones(ishape)
B2 = sigpy.linop.Multiply(ishape, b2)
C2 = A2 * B2

coords3 = (ishape[0] / 2) * (2 * cp.random.rand(num_k_space_points, 3) - 1)
A3 = sigpy.linop.NUFFT(ishape, coords3, oversamp=1.25, width=4)
b3 = cp.ones(ishape)
B3 = sigpy.linop.Multiply(ishape, b3)
C3 = A3 * B3

# complete forward operator
A = sigpy.linop.Vstack([C1, C2, C3])

# set up the operator for regularization
G = sigpy.linop.FiniteDifference(ishape, axes=None)

# simulate data
y = A.apply(x)

y += 0.001 * cp.abs(
    y.max()) * (cp.random.randn(*y.shape) + 1j * cp.random.randn(*y.shape))

proxg = sigpy.prox.L1Reg(G.oshape, lamda=beta)  # define proximal operator

alg = sigpy.app.LinearLeastSquares(A, y, G=G, proxg=proxg)
x_hat = alg.run()

vi = pv.ThreeAxisViewer(np.abs(cp.asnumpy(x_hat)))