"""script to regrid 3D TPI (twisted projection) k-space data"""

import numpy as np
import math
import numpy.typing as npt
from numba import jit


@jit(nopython=True)
def regrid_tpi_data(matrix_size: int,
                    data: npt.NDArray,
                    Nmax: int,
                    kx: npt.NDArray,
                    ky: npt.NDArray,
                    kz: npt.NDArray,
                    kmax: float,
                    kp: float,
                    window: npt.NDArray,
                    cutoff: float,
                    output: npt.NDArray,
                    output_weights=False) -> None:
    """ function to regrid 3D TPI (twisted projection) k-space data on regular k-space grid
        using tri-linear interpolation, correction for sampling density and windowing

    Parameters
    ----------
    matrix_size : int
        size of the k-space grid (matrix)
    data : npt.NDArray
        1D numpy array with the complex k-space data values
    Nmax : int
        process only the first Nmax data points
    kx : npt.NDArray
        1D numpy array containing the x-component of the k-space coordinates
        units ???
    ky : npt.NDArray
        1D numpy array containing the y-component of the k-space coordinates
        units ???
    kz : npt.NDArray
        1D numpy array containing the z-component of the k-space coordinates
        units ???
    kmax : float
        kmax value
    kp : float
        Euclidean distance in k-space until sampling is "radial" with sampling
        density 1/k**2
    window : npt.NDArray
        1D array containing the window
        The window is indexed at: int((|k| / kmax) * cutoff)
    cutoff : float
        Cutoff parameter for window index
    output : npt.NDArray
        3D complex numpy array for the output
        if output_weights is True, it can be real
    output_weights: bool
        output the regridding weights instead of the regridded data
    """
    for i in range(Nmax):
        # Euclidean distance from center of kspace points
        abs_k = np.sqrt(kx[i]**2 + ky[i]**2 + kz[i]**2)

        if abs_k <= kmax:
            # calculate the sampling density which is prop. to abs_k**2
            # in the inner part where sampling is radial
            # and constant in the outer part where the trajectories start
            # twisting
            if abs_k < kp:
                sampling_density = abs_k**2
            else:
                sampling_density = kp**2

            # calculate the window index and value
            i_window = int((abs_k / kmax) * cutoff)

            # shift the kspace coordinates by half the matrix size to get the
            # origin k = (0,0,0) in the center of the array

            kx_shifted = kx[i] + 0.5 * matrix_size
            ky_shifted = ky[i] + 0.5 * matrix_size
            kz_shifted = kz[i] + 0.5 * matrix_size

            # calculate the distances between the lower / upper cells
            # in the kspace grid
            kx_shifted_low = math.floor(kx_shifted)
            ky_shifted_low = math.floor(ky_shifted)
            kz_shifted_low = math.floor(kz_shifted)

            kx_shifted_high = kx_shifted_low + 1
            ky_shifted_high = ky_shifted_low + 1
            kz_shifted_high = kz_shifted_low + 1

            dkx = kx_shifted - kx_shifted_low
            dky = ky_shifted - ky_shifted_low
            dkz = kz_shifted - kz_shifted_low

            if output_weights:
                windowed_data = 1.
            else:
                windowed_data = data[i] * window[i_window]

            # fill the weights array
            output[kx_shifted_low, ky_shifted_low,
                   kz_shifted_low] += (1 - dkx) * (1 - dky) * (
                       1 - dkz) * sampling_density * windowed_data
            output[kx_shifted_high, ky_shifted_low, kz_shifted_low] += (
                dkx) * (1 - dky) * (1 - dkz) * sampling_density * windowed_data
            output[kx_shifted_low, ky_shifted_high, kz_shifted_low] += (
                1 - dkx) * (dky) * (1 - dkz) * sampling_density * windowed_data
            output[kx_shifted_high, ky_shifted_high, kz_shifted_low] += (
                dkx) * (dky) * (1 - dkz) * sampling_density * windowed_data

            output[kx_shifted_low, ky_shifted_low, kz_shifted_high] += (
                1 - dkx) * (1 - dky) * (dkz) * sampling_density * windowed_data
            output[kx_shifted_high, ky_shifted_low, kz_shifted_high] += (
                dkx) * (1 - dky) * (dkz) * sampling_density * windowed_data
            output[kx_shifted_low, ky_shifted_high, kz_shifted_high] += (
                1 - dkx) * (dky) * (dkz) * sampling_density * windowed_data
            output[kx_shifted_high, ky_shifted_high, kz_shifted_high] += (
                dkx) * (dky) * (dkz) * sampling_density * windowed_data


#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
# input variables

if __name__ == '__main__':
    matrix_size: int = 32

    k_spoke = np.linspace(0, 0.4 * matrix_size, 1024)
    n_spokes = 20000

    kx = np.zeros((n_spokes, k_spoke.shape[0]))
    ky = np.zeros((n_spokes, k_spoke.shape[0]))
    kz = np.zeros((n_spokes, k_spoke.shape[0]))

    costheta = 2 * np.random.rand(n_spokes) - 1
    phi = 2 * np.pi * np.random.rand(n_spokes)

    for i in range(n_spokes):
        kx[i, :] = k_spoke * np.sin(phi[i]) * np.sqrt(1 - costheta[i]**2)
        ky[i, :] = k_spoke * np.cos(phi[i]) * np.sqrt(1 - costheta[i]**2)
        kz[i, :] = k_spoke * costheta[i]

    kx = kx.ravel()
    ky = ky.ravel()
    kz = kz.ravel()

    # distance in k-space where trajectories start twisting
    kp: float = 1.5 * k_spoke.max()

    # max k-space distance
    kmax: float = 32.

    cutoff: float = 1.
    window: npt.NDArray = np.ones(100)

    #---------------------------------------------------------------
    # allocated memory for output arrays

    data = np.random.rand(kx.shape[0])

    output: npt.NDArray = np.zeros((matrix_size, matrix_size, matrix_size),
                                   dtype=float)
    #---------------------------------------------------------------

    regrid_tpi_data(matrix_size,
                    data,
                    kx.shape[0],
                    kx,
                    ky,
                    kz,
                    kmax,
                    kp,
                    window,
                    cutoff,
                    output,
                    output_weights=True)

    import pymirc.viewer as pv
    from scipy.ndimage import gaussian_filter
    vi = pv.ThreeAxisViewer([output, gaussian_filter(output, 2.5)])