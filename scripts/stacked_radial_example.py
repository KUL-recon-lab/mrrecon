"""demo to show how to simulate non-uniform kspace data and on how to reconstruct them via conjugate gradient"""

import numpy as np
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt

from mrrecon.linearoperators import GradientOperator
from mrrecon.mroperators import MultiChannelStackedNonCartesianMRAcquisitionModel
from mrrecon.kspace_trajectories import radial_2d_golden_angle
from mrrecon.functionals import SquaredL2Norm

# the reconstruction shape, can be (1,X,X) for 2D tests
recon_shape = (4, 256, 256)
# maximum number of conjugate gradient iterations
num_iterations = 50
# radial undersampling factor
undersampling_factor = 8
# weight of quadratic prior
quadratic_prior_weight = 1e-4
# noise level
noise_level = 0
# gtol parameter for fmin_cg (stops when norm of graient is below gtol)
# default is 1e-5, we just lower it to see what happens if we iterate "long"
gtol = 1e-8

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

num_spokes = int(recon_shape[1] * np.pi / 2) // undersampling_factor
num_samples_per_spoke = recon_shape[0]
print(f'number of spokes {num_spokes}')

print('loading image')
x = np.load('../data/xcat_vol.npz')['arr_0'].reshape(1024, 1024,
                                                     1024).astype(np.float32)

# normalize x to have max 1
x /= x.max()

# swap axes to have sagittal axis in front
x = np.swapaxes(x, 0, 2)

# select recon_shape subvolume from total volume
start0 = x.shape[0] // 2 - recon_shape[0] // 2
end0 = start0 + recon_shape[0]

x = x[start0:end0, ::(x.shape[1] // recon_shape[1]), ::(x.shape[2] //
                                                        recon_shape[2])]

x = x.astype(np.complex64)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
print('setting up operators')

# setup the kspace sample points for the operator acting on the high res image
# here we only take the center which is why kmax is only a fraction np.pi
# reconstructing on a lower res grid needs only the "center part" of kspace

kspace_points = radial_2d_golden_angle(num_spokes, recon_shape[-1])

sens = np.expand_dims(np.ones(recon_shape).astype(x.dtype), 0)

data_operator = MultiChannelStackedNonCartesianMRAcquisitionModel(
    recon_shape, sens, kspace_points, device_number=0)

# set global post scale of data operator such that data_operator.norm()
# is approx 1 (optional but useful e.g. PDHG)
data_operator.post_scale = 1 / 10000.

data = data_operator.forward(x)

if noise_level > 0:
    data += noise_level * (np.random.rand(*data.shape) +
                           1j * np.random.rand(*data.shape))

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# setup the cost functions
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# the data distance is the SquaredL2Norm(exp_data - measured_data)
# we setup a SquaredL2Norm functional "shifted" by the data
data_distance = SquaredL2Norm(xp=np)
data_distance.shift = data

# the actual data fidelity as a function of the image x (not expecte_data(x))
# is the data_distance evaluated at (data_operator.forward(x))
data_fidelity = lambda z: data_distance(
    data_operator.forward(data_operator.unravel_pseudo_complex(z)))

# using the chain rule, we can calculate the gradient of the
# data fidelity term with respect to the image x which is given by
# data_operator.adjoint( data_distance.gradient( data_operator.forward(x) ) )
data_fidelity_gradient = lambda z: data_operator.ravel_pseudo_complex(
    data_operator.adjoint(
        data_distance.gradient(
            data_operator.forward(data_operator.unravel_pseudo_complex(z)))))

# since scipy's optimization algorithms (e.g. fmin_cg) can only handle "flat"
# real input arrays, we use the "(un)ravel complex" methods of the linear
# operator class to transform flattened real arrays into unflattened complex arrays

if quadratic_prior_weight > 0:
    # we setup a simple quadratic prior applied to the gradient of (x)
    prior_operator = GradientOperator(recon_shape, xp=np, dtype=x.dtype)

    prior_norm = SquaredL2Norm(xp=np)
    # the prior weight can be set by setting the "scale" property of the functional
    prior_norm.scale = quadratic_prior_weight

    prior = lambda z: prior_norm(
        prior_operator.forward(prior_operator.unravel_pseudo_complex(z)))
    prior_gradient = lambda z: prior_operator.ravel_pseudo_complex(
        prior_operator.adjoint(
            prior_norm.gradient(
                prior_operator.forward(prior_operator.unravel_pseudo_complex(z)
                                       ))))

    # combine data_fidelity and prior into a single cost function + gradient
    loss = lambda z: data_fidelity(z) + prior(z)
    loss_gradient = lambda z: data_fidelity_gradient(z) + prior_gradient(z)
else:
    loss = lambda z: data_fidelity(z)
    loss_gradient = lambda z: data_fidelity_gradient(z)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# setup a loss function and run a cg reconstruction
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
print('running the recon')

x0 = np.zeros(2 * x.size, dtype=x.real.dtype)

res = fmin_cg(loss,
              x0,
              fprime=loss_gradient,
              maxiter=num_iterations,
              retall=True,
              gtol=gtol)
recon = data_operator.unravel_pseudo_complex(res[0])

# calculate the loss at all iterations
loss_values = [loss(r) for r in res[1]]

# unravel all the recon at every iteration
intermediate_recons = [data_operator.unravel_pseudo_complex(r) for r in res[1]]

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# visualizations
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

ims = dict(cmap=plt.cm.Greys_r, origin='lower', vmin=0, vmax=x.real.max())
fig, ax = plt.subplots(2, 3, figsize=(3 * 4, 2 * 4), sharex=True, sharey=True)
ax[0, 0].imshow(np.abs(x[0, ...]).T, **ims)
ax[1, 0].imshow(np.abs(recon[0, ...]).T, **ims)
ax[0, 0].set_title('slice 0')
ax[0, 1].imshow(np.abs(x[x.shape[0] // 2, ...]).T, **ims)
ax[1, 1].imshow(np.abs(recon[recon_shape[0] // 2, ...]).T, **ims)
ax[0, 1].set_title(f'slice {recon_shape[0] // 2}')
ax[0, 2].imshow(np.abs(x[-1, ...]).T, **ims)
ax[1, 2].imshow(np.abs(recon[-1, ...]).T, **ims)
ax[0, 2].set_title(f'slice {recon_shape[0]-1}')
fig.tight_layout()
fig.show()

# plot the cost function
fig2, ax2 = plt.subplots()
ax2.loglog(np.arange(len(loss_values)), loss_values)
ax2.set_xlabel('iteration')
ax2.set_ylabel('cost function')
ax2.set_title('cost function (loglog plot)')
ax2.grid(ls=':')
fig2.tight_layout()
fig2.show()

# show a few intermetiate reconstructions to visualize convergence
num_intermed = 21
num_rows = int(np.sqrt(num_intermed * 9 / 16))
num_cols = int(np.ceil(num_intermed / num_rows))
fig3, ax3 = plt.subplots(num_rows,
                         num_cols,
                         figsize=(num_cols * 2.7, num_rows * 2.7),
                         sharex=True,
                         sharey=True)
sl = x.shape[0] // 2
its = np.arange(num_intermed) * (len(intermediate_recons) // num_intermed)
its[:4] = its[:4]
its[-1] = len(intermediate_recons) - 1
for i, it in enumerate(its):
    ax3.ravel()[i].imshow(np.abs(intermediate_recons[it][sl, ...]).T, **ims)
    ax3.ravel()[i].set_title(f'it {it}, loss {loss_values[it]:.2e}',
                             fontsize='small')
for axx in ax3.ravel():
    axx.set_axis_off()
fig3.tight_layout()
fig3.show()