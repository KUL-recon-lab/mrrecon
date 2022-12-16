"""generic linear operators related to imaging problems"""
import types
import numpy as np
import numpy.typing as npt

try:
    import cupy.typing as cpt
except:
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
    def x(self) -> npt.NDArray:
        return self._x

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def k(self) -> npt.NDArray:
        return self.xp.fft.fftfreq(self.x.size, d=self.dx) * 2 * self.xp.pi

    @property
    def k_scaled(self) -> npt.NDArray:
        return self.k * self.dx

    @property
    def phase_factor(self) -> npt.NDArray:
        return self._phase_factor

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        return self.xp.fft.fft(
            x, norm='ortho') * self.phase_factor * self.scale_factor

    def adjoint(self, y: npt.NDArray) -> npt.NDArray:
        return self.xp.fft.ifft(y * self.scale_factor *
                                self.xp.conj(self.phase_factor),
                                norm='ortho')

    def inverse(self, y: npt.NDArray) -> npt.NDArray:
        return self.xp.fft.ifft(y / (self.phase_factor * self.scale_factor),
                                norm='ortho')
