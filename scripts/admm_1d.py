"""demo to show how to use ADMM to minimize f(x) + g(Kx) 
   where f(x) is the smooth MR data fidelity with known gradient
   K is the gradient operator
   and g(.) is a gradient norm (e.g. non-smooth L1L2norm -> Total Variation)
"""

import numpy as np
import matplotlib.pyplot as plt

from mrrecon.linearoperators import GradientOperator
from mrrecon.mroperators import FFT1D
from mrrecon.functionals import SquaredL2Norm, L2L1Norm
from mrrecon.algorithms import ADMM

# maximum number of outer ADMM iterations
num_outer_iterations = 500
# maximum number of conjugate gradient iterations
max_num_cg_iterations = 100

# prior norm and weight
prior_norm = L2L1Norm(xp=np)
prior_norm.scale = 3e-1

#prior_norm = SquaredL2Norm(xp=np)
#prior_norm.scale = 3e0

# image size
n = 32
# rho parameter of ADMM
rhos = [1e-1, 1e0, 1e1, 1e2]

noise_level = 0.1
seed = 1

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#--- setup a test image ------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
np.random.seed(seed)

# setup a complex test image
true_img = np.zeros(n, dtype=np.complex128)
true_img[(n // 4):((3 * n) // 4)] = 1

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#--- setup the operators and functionals -------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

sampling_points = np.linspace(-1, 1, n, endpoint=False)

# we setup a SquaredL2Norm functional "shifted" by the data
data_operator = FFT1D(sampling_points, xp=np)
data_distance = SquaredL2Norm(xp=np)

prior_operator = GradientOperator(true_img.shape, xp=np, dtype=true_img.dtype)

# (optional) use a post scaling factor such that the norm of the data and
# prior operator are similar
data_operator.post_scale = (prior_operator.norm() / data_operator.norm())

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#--- generate data -----------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# generate noise free data
noise_free_data = data_operator.forward(true_img)

# add noise to the data
data = noise_free_data + noise_level * (np.random.randn(n) +
                                        1j * np.random.randn(n))

# add the data to the data distance
data_distance.shift = data

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#--- run ADMM updates --------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

recons = np.zeros((len(rhos), n), dtype=complex)
costs = np.zeros((len(rhos), num_outer_iterations))
data_costs = np.zeros((len(rhos), num_outer_iterations))
prior_costs = np.zeros((len(rhos), num_outer_iterations))

for i, rho in enumerate(rhos):
    reconstructor = ADMM(data_operator, data_distance, prior_operator,
                         prior_norm)
    reconstructor.rho = rho
    reconstructor.run(num_outer_iterations, calculate_cost=True)

    recons[i, ...] = reconstructor.x
    costs[i, ...] = reconstructor.cost
    data_costs[i, ...] = reconstructor.cost_data
    prior_costs[i, ...] = reconstructor.cost_prior

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#--- visualizations ----------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

ifft = data_operator.inverse(data)

fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharex='row')
ax[0, 0].plot(true_img.real, 'k')
ax[0, 0].plot(ifft.real, color=plt.cm.tab10(0))
ax[0, 1].plot(true_img.imag, 'k')
ax[0, 1].plot(ifft.imag, color=plt.cm.tab10(0))
ax[0, 2].plot(np.abs(true_img), 'k')
ax[0, 2].plot(np.abs(ifft), color=plt.cm.tab10(0))

for i, rho in enumerate(rhos):
    ax[0, 0].plot(recons[i, ...].real, '--', color=plt.cm.tab10(1 + i))
    ax[0, 1].plot(recons[i, ...].imag, '--', color=plt.cm.tab10(1 + i))
    ax[0, 2].plot(np.abs(recons[i, ...]), '--', color=plt.cm.tab10(1 + i))
    ax[1, 0].loglog(np.arange(1, num_outer_iterations + 1),
                    data_costs[i, ...],
                    color=plt.cm.tab10(1 + i))
    ax[1, 1].loglog(np.arange(1, num_outer_iterations + 1),
                    prior_costs[i, ...],
                    color=plt.cm.tab10(1 + i))
    ax[1, 2].loglog(np.arange(1, num_outer_iterations + 1),
                    costs[i, ...],
                    color=plt.cm.tab10(1 + i),
                    label=f'rho {rho:.1e}')

ax[1, 0].set_title('data fidelity', fontsize='medium')
ax[1, 1].set_title('prior', fontsize='medium')
ax[1, 2].set_title('cost', fontsize='medium')
ax[1, 2].legend()

for axx in ax[0, :]:
    axx.set_ylim(-0.1 * np.abs(ifft.real.max()), 1.1 * ifft.real.max())

for axx in ax.ravel():
    axx.grid(ls=':')

fig.tight_layout()
fig.show()