import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

import mrrecon.algorithms as algorithms
import mrrecon.linearoperators as operators
import mrrecon.functionals as functionals

xp = np
n = 16
noise_level = 0.3
beta = 0.3
niter = 1000
rho = 1.

#------------------------------------------------------------------------

np.random.seed(1)

img = xp.ones((n // 2, n // 2))
img = xp.pad(img, n // 4)

img = img + 1j * img

noisy_img = img + noise_level * xp.random.randn(
    *img.shape) + 1j * noise_level * xp.random.randn(*img.shape)

g_functional = functionals.SquaredL2Norm(xp, shift=noisy_img)
operator = operators.GradientOperator(img.shape, xp=xp, dtype=img.dtype)
f_functional = functionals.L2L1Norm(xp, scale=beta)

operator_norm = np.sqrt(img.ndim * 4)

denoiser = algorithms.PDHG_ALG2(operator,
                                f_functional,
                                g_functional,
                                grad_g_lipschitz=1.,
                                sigma=rho / operator_norm,
                                tau=1 / (rho * operator_norm))

# set initial values for x and xbar
denoiser.x = noisy_img
denoiser.xbar = noisy_img

denoiser.run(niter, calculate_cost=True)

#----------------------------------------------------------------------
# visualizations
pkwargs = dict(vmin=0, vmax=1.2, cmap=plt.cm.Greys)

fig, ax = plt.subplots(2, 4, figsize=(12, 6))
ax[0, 0].imshow(img.real, **pkwargs)
ax[1, 0].imshow(img.imag, **pkwargs)
ax[0, 1].imshow(noisy_img.real, **pkwargs)
ax[1, 1].imshow(noisy_img.imag, **pkwargs)
ax[0, 2].imshow(denoiser.x.real, **pkwargs)
ax[1, 2].imshow(denoiser.x.imag, **pkwargs)
ax[0, 3].loglog(np.arange(1, denoiser.epoch_counter + 1), denoiser.cost)
ax[1, 3].set_axis_off()

ax[0, 0].set_title('noise free image')
ax[0, 1].set_title('noisy image')
ax[0, 2].set_title('PDHG ALG2 denoised')
ax[0, 3].set_title('cost PDHG ALG2')

ax[0, 0].set_ylabel('real part')
ax[1, 0].set_ylabel('imag part')

fig.tight_layout()
fig.show()