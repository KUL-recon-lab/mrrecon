import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

import mrrecon.algorithms as algorithms
import mrrecon.linearoperators as operators
import mrrecon.functionals as functionals

xp = np
n = 16
noise_level = 0.3
beta = 0.2
niter = 1000
rho = 1.

# L1Norm or L2L1Norm
prior_norm = 'L1Norm'

#------------------------------------------------------------------------

np.random.seed(1)

img = xp.ones((n // 2, n // 2))
img = xp.pad(img, n // 4)

noisy_img = img + noise_level * xp.random.randn(*img.shape)

g_functional = functionals.SquaredL2Norm(xp, shift=noisy_img)

operator = operators.GradientOperator(img.shape, xp=xp, dtype=img.dtype)

if prior_norm == 'L1Norm':
    f_functional = functionals.L1Norm(xp, scale=beta)
    grad_g_lipschitz = None
elif prior_norm == 'L2L1Norm':
    f_functional = functionals.L2L1Norm(xp, scale=beta)
    grad_g_lipschitz = 1
else:
    raise ValueError

operator_norm = np.sqrt(img.ndim * 4)

denoiser = algorithms.PDHG_ALG12(operator,
                                 f_functional,
                                 g_functional,
                                 grad_g_lipschitz=grad_g_lipschitz,
                                 sigma=0.99 * rho / operator_norm,
                                 tau=0.99 / (rho * operator_norm))

# set initial values for x and xbar
denoiser.x = noisy_img
denoiser.xbar = noisy_img

denoiser.run(niter, calculate_cost=True)

cost = lambda x: f_functional(operator.forward(x.reshape(img.shape))
                              ) + g_functional(x.reshape(img.shape))

# run brute force minimzation to check if there is a better solution
brute_force_res = minimize(cost, denoiser.x.flatten())

#----------------------------------------------------------------------
# visualizations
pkwargs = dict(vmin=0, vmax=1.2, cmap=plt.cm.Greys)

fig, ax = plt.subplots(2, 3, figsize=(9, 6))
ax[0, 0].imshow(img, **pkwargs)
ax[0, 1].imshow(noisy_img, **pkwargs)
ax[0, 2].imshow(denoiser.x, **pkwargs)
ax[1, 0].imshow(brute_force_res.x.reshape(img.shape), **pkwargs)
ax[1, 1].imshow(denoiser.x - brute_force_res.x.reshape(img.shape),
                vmin=-1e-3,
                vmax=1e-3,
                cmap=plt.cm.bwr)
ax[1, 2].loglog(np.arange(1, denoiser.epoch_counter + 1), denoiser.cost)
ax[1, 2].axhline(brute_force_res.fun, color='k', linewidth=0.5)
ax[1, 2].grid(ls=':')

ax[0, 0].set_title('noise free image')
ax[0, 1].set_title('noisy image')
ax[0, 2].set_title('PDHG ALG denoised')
ax[1, 0].set_title('brute force denoised')
ax[1, 1].set_title('difference')
ax[1, 2].set_title('cost PDHG ALG')

fig.tight_layout()
fig.show()
