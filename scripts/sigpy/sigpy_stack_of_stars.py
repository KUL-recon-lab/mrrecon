"""demo script to show how to setup a stack of radial stars operator in sigpy"""

import numpy as np
import sigpy


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

    ops = img_shape[0] * [rs_out * nufft_op * rs_in]

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

    img_shape = (32, 128, 128)
    num_spokes = 16
    num_points = 256

    trans_fov_cm = 40.

    #-----------------------------------------------------
    # max k value according to Nyquist
    kmax_1_cm = 1. / (2 * (trans_fov_cm / img_shape[1]))

    # generate a random test image
    # for faster execution on a GPU change this to a cupy array
    x = np.random.rand(*img_shape)

    # setup a 2D coordinates for the NUFFTs
    # sigpy needs the coordinates without units
    # ranging from -N/2 ... N/2 if we are at Nyquist
    # if the k-space coordinates have physical units (1/cm)
    # we have to multiply the the FOV (in cm)
    kspace_coords_2d = golden_angle_2d_readout(kmax_1_cm * trans_fov_cm,
                                               num_spokes, num_points)

    #import matplotlib.pyplot as plt
    #plt.ion()
    #plt.plot(kspace_coords_2d[..., 0].T, kspace_coords_2d[..., 1].T)

    fwd_op = stacked_nufft_operator(img_shape, kspace_coords_2d.reshape(-1, 2))

    x_fwd = fwd_op(x)