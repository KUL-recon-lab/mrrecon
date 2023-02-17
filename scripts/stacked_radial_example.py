"""demo to show how to simulate non-uniform kspace data and on how to reconstruct them via conjugate gradient"""

import numpy as np
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt

from mrrecon.linearoperators import GradientOperator
from mrrecon.mroperators import MultiChannelStackedNonCartesianMRAcquisitionModel
from mrrecon.kspace_trajectories import radial_2d_golden_angle
from mrrecon.functionals import SquaredL2Norm

import pymirc.viewer as pv

# the reconstruction shape, can be (1,X,X) for 2D tests
recon_shape = (16, 256, 256)
# maximum number of conjugate gradient iterations
num_iterations = 50
# radial undersampling factor
undersampling_factor = 16
# weight of quadratic prior
quadratic_prior_weight = 0  # 1e-4
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

# swap axes to have sagittal axis in front
x = np.swapaxes(x, 0, 2)

# downsample the volume to 256^3
x = x[0::4, :, :] + x[1::4, :, :] + x[2::4, :, :] + x[3::4, :, :]
x = x[:, 0::4, :] + x[:, 1::4, :] + x[:, 2::4, :] + x[:, 3::4, :]
x = x[:, :, 0::4] + x[:, :, 1::4] + x[:, :, 2::4] + x[:, :, 3::4]

# normalize x to have max 1
x /= x.max()

# cast to complex
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

sens = np.expand_dims(np.ones(x.shape).astype(x.dtype), 0)

data_operator = MultiChannelStackedNonCartesianMRAcquisitionModel(
    x.shape, sens, kspace_points, device_number=0)

# generate "oversampled" data on a has 256 k-space points in the "axial direction"
oversampled_data = data_operator.forward(x)

# select only the "inner" part of the kspace data in the "axial direction"
oversampling_factor = x.shape[0] // recon_shape[0]

start0 = x.shape[0] * (oversampling_factor - 1) // (2 * oversampling_factor)
end0 = x.shape[0] * (oversampling_factor + 1) // (2 * oversampling_factor)

data = oversampled_data[:, start0:end0, :]

# we have to divide by the sqrt of the oversampling factor because we use
# norm = 'ortho' in the fft
data /= np.sqrt(oversampling_factor)

if noise_level > 0:
    data += noise_level * (np.random.rand(*data.shape) +
                           1j * np.random.rand(*data.shape))

# setup a new operator that corresponds to the downsampled (inner part of the) data
sens = np.expand_dims(np.ones(recon_shape).astype(x.dtype), 0)

data_operator = MultiChannelStackedNonCartesianMRAcquisitionModel(
    recon_shape, sens, kspace_points, device_number=0)

# estimate the norm of the forward operator
op_norm = data_operator.norm(num_iter=5)

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

x0 = data_operator.ravel_pseudo_complex(
    data_operator.adjoint(data) / (op_norm**2))

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
intermediate_recons = np.array(
    [np.abs(data_operator.unravel_pseudo_complex(r)) for r in res[1]])

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# visualizations
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# "upsample" the recon to the original 256x256x256 grid for visualization
tmp = oversampling_factor // 2
upsampled_recon = np.pad(
    np.repeat(np.abs(recon), oversampling_factor, axis=0)[tmp:, :, :],
    ((0, tmp), (0, 0), (0, 0)))

ims = dict(cmap=plt.cm.Greys_r, vmin=0, vmax=1.2 * x.real.max())
vi = pv.ThreeAxisViewer([upsampled_recon, x.real], imshow_kwargs=ims)

# plot the cost function
fig2, ax2 = plt.subplots()
ax2.loglog(np.arange(len(loss_values)), loss_values)
ax2.set_xlabel('iteration')
ax2.set_ylabel('cost function')
ax2.set_title('cost function (loglog plot)')
ax2.grid(ls=':')
fig2.tight_layout()
fig2.show()