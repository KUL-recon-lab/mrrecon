import abc
import numpy as np
import numpy.typing as npt

try:
    import cupy.typing as cpt
except ModuleNotFoundError:
    import numpy.typing as cpt


class DiffMetric(abc.ABC):

    def __init__(self,
                 y: npt.NDArray | cpt.NDArray,
                 weights=None,
                 xp=np) -> None:
        self._y = y
        self._weights = None
        self._xp = xp

    @abc.abstractmethod
    def _call_from_diff(self, d: npt.NDArray | cpt.NDArray) -> float:
        raise NotImplementedError

    def _diff(self, x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return x - self._y

    def __call__(self, x: npt.NDArray | cpt.NDArray) -> float:
        diff = self._diff(x)
        if self._weights is not None:
            diff *= self._weights

        return self._call_from_diff(diff)


class MSE(DiffMetric):

    def _call_from_diff(self, d: npt.NDArray | cpt.NDArray) -> float:
        return (self._xp.conj(d) * d).sum().real


class MAE(DiffMetric):

    def _call_from_diff(self, d: npt.NDArray | cpt.NDArray) -> float:
        return self._xp.abs(d).sum()
