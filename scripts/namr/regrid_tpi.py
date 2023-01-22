"""script to regrid 3D TPI (twisted projection) k-space data"""

import numpy as np
import math
import numpy.typing as npt
from numba import jit
import matplotlib.pyplot as plt


@jit(nopython=True)
def regrid_tpi_data(matrix_size: int,
                    delta_k: float,
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
    delta_k : float
        the k-space spacing
    data : npt.NDArray
        1D numpy array with the complex k-space data values
    Nmax : int
        process only the first Nmax data points
    kx : npt.NDArray
        1D numpy array containing the x-component of the k-space coordinates
        unit 1/cm
    ky : npt.NDArray
        1D numpy array containing the y-component of the k-space coordinates
        unit 1/cm
    kz : npt.NDArray
        1D numpy array containing the z-component of the k-space coordinates
        unit 1/cm
    kmax : float
        kmax value
        unit 1/cm
    kp : float
        Euclidean distance in k-space until sampling is "radial" with sampling
        density 1/k**2
        unit 1/cm
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
            # we also divide by delta k, such that we get k in k_space grid units

            kx_shifted = (kx[i] / delta_k) + 0.5 * (matrix_size)
            ky_shifted = (ky[i] / delta_k) + 0.5 * (matrix_size)
            kz_shifted = (kz[i] / delta_k) + 0.5 * (matrix_size)

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
                windowed_data = 1. + 0j
            else:
                windowed_data = data[i] * window[i_window]

            if (kx_shifted_low >= 0) and (ky_shifted_low >=
                                          0) and (kz_shifted_low >= 0):
                # fill output array according to trilinear interpolation
                output[kx_shifted_low, ky_shifted_low,
                       kz_shifted_low] += (1 - dkx) * (1 - dky) * (
                           1 - dkz) * sampling_density * windowed_data
                output[kx_shifted_high, ky_shifted_low,
                       kz_shifted_low] += (dkx) * (1 - dky) * (
                           1 - dkz) * sampling_density * windowed_data
                output[kx_shifted_low, ky_shifted_high,
                       kz_shifted_low] += (1 - dkx) * (dky) * (
                           1 - dkz) * sampling_density * windowed_data
                output[kx_shifted_high, ky_shifted_high, kz_shifted_low] += (
                    dkx) * (dky) * (1 - dkz) * sampling_density * windowed_data

                output[kx_shifted_low, ky_shifted_low,
                       kz_shifted_high] += (1 - dkx) * (1 - dky) * (
                           dkz) * sampling_density * windowed_data
                output[kx_shifted_high, ky_shifted_low, kz_shifted_high] += (
                    dkx) * (1 - dky) * (dkz) * sampling_density * windowed_data
                output[kx_shifted_low, ky_shifted_high, kz_shifted_high] += (
                    1 - dkx) * (dky) * (dkz) * sampling_density * windowed_data
                output[kx_shifted_high, ky_shifted_high, kz_shifted_high] += (
                    dkx) * (dky) * (dkz) * sampling_density * windowed_data


def read_single_tpi_gradient_file(gradient_file: str,
                                  gamma_by_2pi: float = 1126.2,
                                  num_header_elements: int = 6):

    header = np.fromfile(gradient_file,
                         dtype=np.int16,
                         offset=0,
                         count=num_header_elements)

    # number of cones
    num_cones = int(header[0])
    # number of points in a single readout
    num_points = int(header[1])

    # time sampling step in seconds
    dt = float(header[2]) * (1e-6)

    # maximum gradient strength in G/cm corresponds to max short value (2**15 - 1 = 32767
    max_gradient = float(header[3]) / 100

    # number of readouts per cone
    num_readouts_per_cone = np.fromfile(gradient_file,
                                        dtype=np.int16,
                                        offset=num_header_elements * 2,
                                        count=num_cones)

    gradient_array = np.fromfile(gradient_file,
                                 dtype=np.int16,
                                 offset=(num_header_elements + num_cones) * 2,
                                 count=num_cones * num_points).reshape(
                                     num_cones, num_points)

    # calculate k_array in (1/cm)
    k_array = np.cumsum(
        gradient_array,
        axis=1) * dt * gamma_by_2pi * max_gradient / (2**15 - 1)

    return k_array, header, num_readouts_per_cone


def read_tpi_gradient_files(file_base: str,
                            x_suffix: str = 'x.grdb',
                            y_suffix: str = 'y.grdb',
                            z_suffix: str = 'z.grdb',
                            **kwargs):

    kx, header, num_readouts_per_cone = read_single_tpi_gradient_file(
        f'{file_base}.{x_suffix}', **kwargs)
    ky, header, num_readouts_per_cone = read_single_tpi_gradient_file(
        f'{file_base}.{y_suffix}', **kwargs)
    kz, header, num_readouts_per_cone = read_single_tpi_gradient_file(
        f'{file_base}.{z_suffix}', **kwargs)

    kx_rotated = np.zeros((num_readouts_per_cone.sum(), kx.shape[1]))
    ky_rotated = np.zeros((num_readouts_per_cone.sum(), ky.shape[1]))
    kz_rotated = np.zeros((num_readouts_per_cone.sum(), kz.shape[1]))

    num_readouts_cumsum = np.cumsum(
        np.concatenate(([0], num_readouts_per_cone)))

    # start angle of first readout in each cone
    phi0s = np.linspace(0, 2 * np.pi, kx.shape[0], endpoint=False)

    for i_cone in range(header[0]):
        num_readouts = num_readouts_per_cone[i_cone]

        phis = np.linspace(phi0s[i_cone],
                           2 * np.pi + phi0s[i_cone],
                           num_readouts,
                           endpoint=False)

        for ir in range(num_readouts):
            kx_rotated[ir + num_readouts_cumsum[i_cone], :] = np.cos(
                phis[ir]) * kx[i_cone, :] - np.sin(phis[ir]) * ky[i_cone, :]
            ky_rotated[ir + num_readouts_cumsum[i_cone], :] = np.sin(
                phis[ir]) * kx[i_cone, :] + np.cos(phis[ir]) * ky[i_cone, :]
            kz_rotated[ir + num_readouts_cumsum[i_cone], :] = kz[i_cone, :]

    return kx_rotated, ky_rotated, kz_rotated, header, num_readouts_per_cone


def show_tpi_readout(kx,
                     ky,
                     kz,
                     header,
                     num_readouts_per_cone,
                     start_cone=0,
                     end_cone=None,
                     cone_step=2,
                     readout_step=6,
                     step=20):
    num_cones = header[0]

    if end_cone is None:
        end_cone = num_cones

    cone_numbers = np.arange(start_cone, end_cone, cone_step)

    # cumulative sum of readouts per cone
    rpc_cumsum = np.concatenate(([0], num_readouts_per_cone.cumsum()))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    for ic in cone_numbers:
        ax.scatter3D(
            kx[rpc_cumsum[ic]:rpc_cumsum[ic + 1]:readout_step, ::step],
            ky[rpc_cumsum[ic]:rpc_cumsum[ic + 1]:readout_step, ::step],
            kz[rpc_cumsum[ic]:rpc_cumsum[ic + 1]:readout_step, ::step],
            s=0.5)

    ax.set_xlim(kx.min(), kx.max())
    ax.set_ylim(ky.min(), ky.max())
    ax.set_zlim(kz.min(), kz.max())
    fig.tight_layout()
    fig.show()


#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

if __name__ == '__main__':

    kx, ky, kz, header, n_readouts_per_cone = read_tpi_gradient_files(
        '/data/tpi_gradients/n28p4dt10g16_23Na_v1')
    #    '/data/tpi_gradients/n28p4dt10g32_23Na_v0')

    #show_tpi_readout(kx, ky, kz, header, n_readouts_per_cone)

    matrix_size: int = 64
    field_of_view: float = 22.

    kp: float = 0.4 * 18 / field_of_view
    kmax: float = 1.8 * 18 / field_of_view

    cutoff: float = 1.
    window: npt.NDArray = np.ones(100)

    #---------------------------------------------------------------
    # allocated memory for output arrays

    data = np.random.rand(kx.shape[0]) + 1j * np.random.rand(kx.shape[0])

    output: npt.NDArray = np.zeros((matrix_size, matrix_size, matrix_size),
                                   dtype=complex)
    #---------------------------------------------------------------

    regrid_tpi_data(matrix_size,
                    1 / field_of_view,
                    data,
                    kx.size,
                    kx.ravel(),
                    ky.ravel(),
                    kz.ravel(),
                    0.95 * kmax,
                    kp,
                    window,
                    cutoff,
                    output,
                    output_weights=True)

    regrid_tpi_data(matrix_size,
                    1 / field_of_view,
                    data,
                    kx.size,
                    -kx.ravel(),
                    -ky.ravel(),
                    -kz.ravel(),
                    0.95 * kmax,
                    kp,
                    window,
                    cutoff,
                    output,
                    output_weights=True)

    import pymirc.viewer as pv
    vi = pv.ThreeAxisViewer(output.real)
