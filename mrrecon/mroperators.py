"""generic linear operators related to imaging problems"""
import types
import numpy as np
import numpy.typing as npt
import pynufft

try:
    import cupy.typing as cpt
except ModuleNotFoundError:
    import numpy.typing as cpt

from .linearoperators import LinearOperator


class FFT1D(LinearOperator):

    def __init__(self,
                 x: npt.NDArray | cpt.NDArray,
                 xp: types.ModuleType = np) -> None:
        """ 1D fast fourier transform operator matched to reproduce results of symmetric continous FT

        Parameters
        ----------
        x : npt.NDArray | cpt.NDArray
            spatial sampling points
        xp : types.ModuleType, optional
            array model to use, by default numpy
        """
        super().__init__(input_shape=x.shape,
                         output_shape=x.shape,
                         xp=xp,
                         input_dtype=complex,
                         output_dtype=complex)

        self._dx = float(x[1] - x[0])
        self._x = x
        self._phase_factor = self.dx * self.xp.exp(-1j * self.k * float(x[0]))
        self._scale_factor = float(self.xp.sqrt(x.size / (2 * self.xp.pi)))

    @property
    def x(self) -> npt.NDArray | cpt.NDArray:
        return self._x

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def k(self) -> npt.NDArray | cpt.NDArray:
        return self.xp.fft.fftfreq(self.x.size, d=self.dx) * 2 * self.xp.pi

    @property
    def k_scaled(self) -> npt.NDArray | cpt.NDArray:
        return self.k * self.dx

    @property
    def phase_factor(self) -> npt.NDArray | cpt.NDArray:
        return self._phase_factor

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    def _forward(self,
                 x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return self.xp.fft.fft(
            x, norm='ortho') * self.phase_factor * self.scale_factor

    def _adjoint(self,
                 y: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return self.xp.fft.ifft(y * self.scale_factor *
                                self.xp.conj(self.phase_factor),
                                norm='ortho')

    def inverse(self,
                y: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return self.xp.fft.ifft(y / (self.phase_factor * self.scale_factor),
                                norm='ortho')


class T2CorrectedFFT1D(FFT1D):
    """FFT including T2star decay modeling during readout"""

    def __init__(self,
                 x: npt.NDArray,
                 t_readout: npt.NDArray,
                 T2star: npt.NDArray,
                 xp: types.ModuleType = np) -> None:

        super().__init__(x=x, xp=xp)

        self._t_readout = t_readout
        self._T2star = T2star

        # precalculate the decay envelopes at the readout times
        # assumes that readout time is a function of abs(k)
        self._n = self.x.shape[0]
        self._decay_envs = self.xp.zeros((self._n // 2 + 1, self._n))
        self._masks = self.xp.zeros((self._n // 2 + 1, self._n))
        inds = self.xp.where(T2star > 0)
        for i, t in enumerate(t_readout[:(self._n // 2 + 1)]):
            tmp = self.xp.zeros(self._n)
            tmp[inds] = (t / T2star[inds])
            self._decay_envs[i, :] = xp.exp(-tmp)
            self._masks[i, i] = 1
            self._masks[i, -i] = 1

    @property
    def masks(self) -> npt.NDArray | cpt.NDArray:
        return self._masks

    @property
    def decay_envs(self) -> npt.NDArray | cpt.NDArray:
        return self._decay_envs

    def _forward(self,
                 x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        y = self.xp.zeros(self.output_shape, dtype=self.output_dtype)

        for i in range(self._n // 2 + 1):
            y += super()._forward(self.decay_envs[i, :] * x) * self.masks[i, :]

        return y

    def _adjoint(self,
                 y: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        x = self.xp.zeros(self.output_shape, dtype=self.input_dtype)

        for i in range(self._n // 2 + 1):
            x += super()._adjoint(self.masks[i, :] * y) * self.decay_envs[i, :]

        return x


class MultiChannelStackedNonCartesianMRAcquisitionModel(LinearOperator):
    """acquisition model for multi channel MR with non cartesian stacked sampling using pynufft"""

    def __init__(self,
                 input_shape: tuple[int, int, int],
                 coil_sensitivities: npt.NDArray,
                 k_space_sample_points: npt.NDArray,
                 interpolation_size: tuple[int, int] = (6, 6),
                 scaling_factor: float = 1.,
                 device_number=0) -> None:
        """
        Parameters 
        ----------
 
        input_shape : tuple[int, int, int]
            shape of the complex input image
        coil_sensitivities : npt.NDArray
            the complex coil sensitivities, shape (num_channels, image_shape)
        k_space_sample_points, npt.NDArray
            2D coordinates of kspace sample points (same for every stack)
        interpolation_size: tuple(int,int), optional
            interpolation size for nufft, default (6,6)
        scaling_factor: float, optional
            extra scaling factor applied to the adjoint, default 1
        device_number: int, optional
            device from pynuffts device list to use, default 0
        """

        self._coil_sensitivities = coil_sensitivities
        self._num_channels = coil_sensitivities.shape[0]
        self._scaling_factor = scaling_factor

        self._kspace_sample_points = k_space_sample_points

        # size of the oversampled kspace grid
        self._Kd = tuple(2 * x for x in input_shape[1:])
        # the adjoint from pynufft needs to be scaled by this factor
        self._adjoint_scaling_factor = float(np.prod(self._Kd))

        if interpolation_size is None:
            self._interpolation_size = (6, 6)
        else:
            self._interpolation_size = interpolation_size

        self._device_number = device_number
        self._device = pynufft.helper.device_list()[self._device_number]

        # setup a nufft object for every stack
        self._nufft_2d = pynufft.NUFFT(self._device)
        self._nufft_2d.plan(self.kspace_sample_points, input_shape[1:],
                            self._Kd, self._interpolation_size)

        super().__init__(input_shape=input_shape,
                         output_shape=(self._num_channels, input_shape[0],
                                       self._kspace_sample_points.shape[0]),
                         xp=np,
                         input_dtype=np.complex64,
                         output_dtype=np.complex64)

    @property
    def num_channels(self) -> int:
        """
        Returns
        -------
        int
            number of channels (coils)
        """
        return self._num_channels

    @property
    def coil_sensitivities(self) -> npt.NDArray:
        """
        Returns
        -------
        npt.NDArray
            array of coil sensitivities
        """
        return self._coil_sensitivities

    @property
    def kspace_sample_points(self) -> npt.NDArray:
        """
        Returns
        -------
        npt.NDArray
            the kspace sample points
        """
        return self._kspace_sample_points

    @property
    def nufft_2d(self) -> pynufft.NUFFT:
        return self._nufft_2d

    @property
    def scaling_factor(self) -> float:
        return self._scaling_factor

    @property
    def adjoint_scaling_factor(self) -> float:
        return self._adjoint_scaling_factor

    def _forward(self, x: npt.NDArray) -> npt.NDArray:

        y = self.xp.zeros(self.output_shape, dtype=self.output_dtype)

        for i in range(self._num_channels):
            # perform a 1D FFT along the "stack axis"
            tmp = self.xp.fft.fftshift(self.xp.fft.fftn(self.xp.fft.fftshift(
                self.coil_sensitivities[i, ...] * x, axes=0),
                                                        axes=[0]),
                                       axes=0)

            # series of 2D NUFFTs
            for k in range(self.input_shape[0]):
                y[i, k, ...] = self.nufft_2d.forward(tmp[k, ...])

        if self._scaling_factor != 1:
            y *= self.scaling_factor

        return y

    def _adjoint(self, y: npt.NDArray) -> npt.NDArray:

        x = self.xp.zeros(self.input_shape, dtype=self.input_dtype)

        for i in range(self._num_channels):
            tmp = self.xp.zeros(self.input_shape, dtype=self.input_dtype)
            # series of 2D adjoint NUFFTs
            for k in range(self.input_shape[0]):
                tmp[k, ...] = self.nufft_2d.adjoint(y[i, k, ...])

            x += self.xp.conj(
                self.coil_sensitivities[i, ...]) * self.xp.fft.ifftshift(
                    self.xp.fft.ifftn(self.xp.fft.ifftshift(tmp, axes=0),
                                      axes=[0]),
                    axes=0)

        # when using numpy's fftn with the default normalization
        # we have to multiply the inverse with input_shape[0] to get the adjoint
        x *= (self.adjoint_scaling_factor * self.input_shape[0])

        if self._scaling_factor != 1:
            x *= self.scaling_factor

        return x
