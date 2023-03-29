"""demo script to show how to solve ADMM subproblem (1) using sigpy"""

#TODO:
# use several 3D images (corresponding to different gates) instead of single 3D image
# -> should be possible using the "diag" operator (as in the stack of stars operator)

import numpy as np
import sigpy
import matplotlib.pyplot as plt


def stacked_nufft_operator(img_shape: tuple, coords: np.ndarray):
    """setup a stacked 2D NUFFT sigpy operator acting on a 3D image
       the opeator first performs a 1D FFT along the "z" axis (0 or left-most axis)
       followed by applying 2D NUFFTS to all "slices"
       
    Parameters
    ----------
        img_shape: tuple
            shape of the image
        coords: array 
            coordinates of the k-space samples
            shape (n_k_space_points,2)
            units: "unitless" -> -N/2 ... N/2 at Nyquist (sigpy convention)

    Returns
    -------
        Diag: a stack of NUFFT operators
    """

    # setup the FFT operator along the "z" axis
    ft0_op = sigpy.linop.FFT(img_shape, axes=(0, ))

    # setup a 2D NUFFT operator for the start
    nufft_op = sigpy.linop.NUFFT(img_shape[1:], coords)

    # reshaping operators that remove / expand dimensions
    rs_in = sigpy.linop.Reshape(img_shape[1:], (1, ) + img_shape[1:])
    rs_out = sigpy.linop.Reshape((1, ) + tuple(nufft_op.oshape),
                                 nufft_op.oshape)

    # setup a list of "n" 2D NUFFT operators
    ops = img_shape[0] * [rs_out * nufft_op * rs_in]

    # apply 2D NUFFTs to all "slices" using the sigpy Diag operator
    return sigpy.linop.Diag(ops, iaxis=0, oaxis=0) * ft0_op


def golden_angle_2d_readout(kmax, num_spokes, num_points):
    tmp = np.linspace(-kmax, kmax, num_points)
    k = np.zeros((num_spokes, num_points, 2))

    ga = np.pi / ((1 + np.sqrt(5)) / 2)

    for i in range(num_spokes):
        phi = (i * ga) % (2 * np.pi)
        k[i, :, 0] = tmp * np.cos(phi)
        k[i, :, 1] = tmp * np.sin(phi)

    return k


if __name__ == '__main__':
    np.random.seed(1)

    img_shape = (6, 32, 32)
    num_spokes = 16
    num_points = 32

    trans_fov_cm = 40.

    # weight of the quadratic penalty term
    lam = 4.2

    #-----------------------------------------------------
    # max k value according to Nyquist
    kmax_1_cm = 1. / (2 * (trans_fov_cm / img_shape[1]))

    # generate 3 random test images
    # for faster execution on a GPU change this to a cupy array
    img1 = np.random.rand(*img_shape) + 1j * np.random.rand(*img_shape)
    img2 = np.random.rand(*img_shape) + 1j * np.random.rand(*img_shape)
    img3 = np.random.rand(*img_shape) + 1j * np.random.rand(*img_shape)

    # stack all 3D images into a 4D array
    img_4d = np.array([img1, img2, img3])

    # setup a random 4D "bias" term for the quadratic penalty
    b_4d = np.random.rand(*img_4d.shape)

    # setup a 2D coordinates for the NUFFTs
    # sigpy needs the coordinates without units
    # ranging from -N/2 ... N/2 if we are at Nyquist
    # if the k-space coordinates have physical units (1/cm)
    # we have to multiply the the FOV (in cm)
    kspace_coords_2d = golden_angle_2d_readout(kmax_1_cm * trans_fov_cm,
                                               num_spokes, num_points)

    # setup the operator that acts on a single 3D image
    fwd_op_3d = stacked_nufft_operator(img_shape,
                                       kspace_coords_2d.reshape(-1, 2))

    # setup the operator that applies the 3d operator to a stack of 3 images
    rs_in = sigpy.linop.Reshape(img_shape, (1, ) + img_shape)
    rs_out = sigpy.linop.Reshape((1, ) + tuple(fwd_op_3d.oshape),
                                 fwd_op_3d.oshape)
    ops = img_4d.shape[0] * [rs_out * fwd_op_3d * rs_in]
    fwd_op_4d = sigpy.linop.Diag(ops, iaxis=0, oaxis=0)

    # generate (noiseless) data
    data_4d = fwd_op_4d(img_4d)

    # setup algorithm to solve ADMM subproblem (1)
    # sigpy's LinearLeastSquares solves:
    # min_x 0.5 * || fwd_op * x - data ||_2^2 + 0.5 * lambda * || x - z ||_2^2
    # https://sigpy.readthedocs.io/en/latest/generated/sigpy.app.LinearLeastSquares.html#sigpy.app.LinearLeastSquares
    # for this problem, sigpy uses conjugate gradient
    x0 = np.zeros_like(img_4d)

    alg = sigpy.app.LinearLeastSquares(fwd_op_4d,
                                       data_4d,
                                       x=x0,
                                       z=b_4d,
                                       lamda=lam,
                                       max_iter=50,
                                       save_objective_values=True)

    # run the algorithm
    res = alg.run()

    # setup the cost function manually to double check that we optimize what we want
    cost = lambda x: 0.5 * (np.abs(fwd_op_4d(x) - data_4d)**2).sum(
    ) + 0.5 * lam * (np.abs(x - b_4d)**2).sum()

    assert (np.isclose(cost(res), alg.objective_values[-1]))

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(kspace_coords_2d[..., 0].T, kspace_coords_2d[..., 1].T)
    ax[1].semilogy(alg.objective_values)
    ax[1].set_ylim(0, max(alg.objective_values[1:]))
    fig.tight_layout()
    fig.show()