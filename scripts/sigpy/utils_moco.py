from __future__ import annotations

import sigpy
import numpy as np
import numpy.typing as npt
import cupy.typing as cpt


def golden_angle_2d_readout(kmax: float, num_spokes: int,
                            num_points: int) -> npt.NDArray:
    """2D golden angle kspace trajectory

    Parameters
    ----------
    kmax : float
        maximum absolute k-space value
    num_spokes : int
        number of spokes (readout lines)
    num_points : int
        number of readout points per spoke

    Returns
    -------
    npt.NDArray
    """
    tmp = np.linspace(-kmax, kmax, num_points)
    k = np.zeros((num_spokes, num_points, 2))

    ga = np.pi / ((1 + np.sqrt(5)) / 2)

    for i in range(num_spokes):
        phi = (i * ga) % (2 * np.pi)
        k[i, :, 0] = tmp * np.cos(phi)
        k[i, :, 1] = tmp * np.sin(phi)

    return k


def stacked_nufft_operator(
        img_shape: tuple,
        coords: npt.NDArray | cpt.NDArray) -> sigpy.linop.Diag:
    """setup a stacked 2D NUFFT sigpy operator acting on a 3D image
       the opeator first performs a 1D FFT along the "z" axis (0 or left-most axis)
       followed by applying 2D NUFFTS to all "slices"
       
    Parameters
    ----------
        img_shape: tuple
            shape of the image
        coords: (numpy or cupy) array 
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
