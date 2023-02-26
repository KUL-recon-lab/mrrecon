"""minimal script that shows how to do gradient descent with box constraints"""
import sigpy
import cupy as cp
import numpy as np

import pymirc.viewer as pv


class MyL2Gradient:

    def __init__(self, A: sigpy.linop.Linop, y: cp.ndarray):
        self._y = y
        self._A = A

    def __call__(self, r: cp.ndarray):
        return self._A.H.apply(self._A.apply(r) - self._y)


#--------------------------------------------------------------------------
ishape = (8, 1)

#--------------------------------------------------------------------------

x = cp.zeros(ishape, dtype=cp.float64)
x[((1 * ishape[0]) // 4):((3 * ishape[0]) // 4)] = 1

# setup a (composite) forward operator
A = sigpy.linop.MatMul(ishape, cp.random.rand(2 * ishape[0], ishape[0]))

# simulate noisefree data
y = A.apply(x)

mygrad = MyL2Gradient(A, y)
proxg = sigpy.prox.BoxConstraint(ishape, 0.05, 0.8)

x0 = cp.random.rand(*ishape)

# estimate the max eigenvalue of A.H*A which is the Lipschitz of the gradient
app = sigpy.app.MaxEig(A.H * A)
gradLip = app.run()

alg = sigpy.alg.GradientMethod(mygrad,
                               x0,
                               1. / gradLip,
                               proxg=proxg,
                               accelerate=False,
                               max_iter=500,
                               tol=0)

while not alg.done():
    alg.update()

print(x.ravel())
print(alg.x.ravel())