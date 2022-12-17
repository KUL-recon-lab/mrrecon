import abc
import types
import functools
import numpy as np
import numpy.typing as npt

try:
    import cupy.typing as cpt
except ModuleNotFoundError:
    import numpy.typing as cpt


class AnalysticalFourierSignal(abc.ABC):
    """abstract base class for 1D signals where the analytical Fourier transform exists"""

    def __init__(self,
                 scale: float = 1.,
                 stretch: float = 1.,
                 shift: float = 0.,
                 xp: types.ModuleType = np,
                 T2star: float = 10.):
        self._scale = scale
        self._stretch = stretch
        self._shift = shift
        self._xp = xp
        self._T2star = T2star

    @property
    def T2star(self) -> float:
        return self._T2star

    @property
    def xp(self) -> types.ModuleType:
        return self._xp

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def stretch(self) -> float:
        return self._stretch

    @property
    def shift(self) -> float:
        return self._shift

    @abc.abstractmethod
    def _signal(self,
                x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        raise NotImplementedError

    @abc.abstractmethod
    def _continous_ft(
            self, k: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        raise NotImplementedError

    def signal(self,
               x: npt.NDArray | cpt.NDArray,
               t: float = 0) -> npt.NDArray | cpt.NDArray:
        return self.xp.exp(-t / self.T2star) * self.scale * self._signal(
            (x - self.shift) * self.stretch)

    def continous_ft(self,
                     k: npt.NDArray | cpt.NDArray,
                     t: float = 0) -> npt.NDArray | cpt.NDArray:
        return self.xp.exp(-t / self.T2star) * self.scale * self.xp.exp(
            -1j * self.shift * k) * self._continous_ft(
                k / self.stretch) / self.xp.abs(self.stretch)


class SquareSignal(AnalysticalFourierSignal):

    def _signal(self,
                x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        y = self.xp.zeros_like(x)
        ipos = self.xp.where(self.xp.logical_and(x >= 0, x < 0.5))
        ineg = self.xp.where(self.xp.logical_and(x >= -0.5, x < 0))

        y[ipos] = 1
        y[ineg] = 1

        return y

    def _continous_ft(
            self, k: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return self.xp.sinc(k / 2 / self.xp.pi) / self.xp.sqrt(2 * self.xp.pi)


class TriangleSignal(AnalysticalFourierSignal):

    def _signal(self,
                x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        y = self.xp.zeros_like(x)
        ipos = self.xp.where(self.xp.logical_and(x >= 0, x < 1))
        ineg = self.xp.where(self.xp.logical_and(x >= -1, x < 0))

        y[ipos] = 1 - x[ipos]
        y[ineg] = 1 + x[ineg]

        return y

    def _continous_ft(
            self, k: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return self.xp.sinc(k / 2 / self.xp.pi)**2 / self.xp.sqrt(
            2 * self.xp.pi)


class GaussSignal(AnalysticalFourierSignal):

    def _signal(self,
                x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return self.xp.exp(-x**2)

    def _continous_ft(
            self, k: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return self.xp.sqrt(self.xp.pi) * self.xp.exp(
            -(k**2) / (4)) / self.xp.sqrt(2 * self.xp.pi)


class CompoundAnalysticalFourierSignal():

    def __init__(self, signals: list[AnalysticalFourierSignal]):
        self._signals = signals

    @property
    def signals(self) -> list[AnalysticalFourierSignal]:
        return self._signals

    def T2star(self,
               x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return functools.reduce(
            lambda a, b: a + b,
            [z.T2star * (z.signal(x, t=0) > 0) for z in self.signals])

    def signal(self,
               x: npt.NDArray | cpt.NDArray,
               t: float = 0) -> npt.NDArray | cpt.NDArray:
        return functools.reduce(lambda a, b: a + b,
                                [z.signal(x, t=t) for z in self.signals])

    def continous_ft(self,
                     x: npt.NDArray | cpt.NDArray,
                     t: float = 0) -> npt.NDArray | cpt.NDArray:
        return functools.reduce(lambda a, b: a + b,
                                [z.continous_ft(x, t=t) for z in self.signals])
