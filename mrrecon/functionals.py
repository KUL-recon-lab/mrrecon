import abc
import types

import numpy as np
import numpy.typing as npt

try:
    import cupy.typing as cpt
except ModuleNotFoundError:
    import numpy.typing as cpt


class Functional(abc.ABC):
    """abstract base class for a functional g(x) =  scale * f(x - shift)"""

    def __init__(self,
                 xp: types.ModuleType = np,
                 scale: float = 1.,
                 shift: float | npt.NDArray | cpt.NDArray = 0.):
        self._xp = xp
        self._scale = scale  # positive scale factor (g(x) = scale * f(x))
        self._shift = shift

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value

    @property
    def shift(self) -> float | npt.NDArray | cpt.NDArray:
        return self._shift

    @shift.setter
    def shift(self, value: float | npt.NDArray | cpt.NDArray) -> None:
        self._shift = value

    @property
    def xp(self) -> types.ModuleType:
        return self._xp

    @abc.abstractmethod
    def _call_f(self, x: npt.NDArray | cpt.NDArray) -> float:
        """f(x)"""
        raise NotImplementedError

    def __call__(self, x: npt.NDArray | cpt.NDArray) -> float:
        """g(x) = scale * f(x - shift)"""
        return self._scale * self._call_f(x - self.shift)


class SmoothFunctional(Functional):
    """smooth functional  g(x) =  scale * f(x - shift) where gradient of f(x) is known"""

    @abc.abstractmethod
    def _gradient_f(self,
                    x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """gradient of f(x)"""
        raise NotImplementedError

    def gradient(self,
                 x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """gradient of g(x)"""
        return self.scale * self._gradient_f(x - self.shift)


class ConvexFunctionalWithProx(Functional):
    """abstract class for functional g(x) =  scale * f(x - shift) 
       where either prox_f^sigma or prox_f*^sigma is known (analytically)

       if either of the two proxes is known, we can calculate prox_g^sigma and prox_g*^sigma
       f*/g* are the convex dual functions of g/f
    """

    @abc.abstractmethod
    def prox(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_g^sigma"""
        raise NotImplementedError

    @abc.abstractmethod
    def prox_convex_dual(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_g*^sigma"""
        raise NotImplementedError


class ConvexFunctionalWithPrimalProx(Functional):
    """abstract class for functional g(x) =  scale * f(x - shift) 
       where prox_f^sigma is known (analytically)

       we can calculate prox_g*^sigma using Moreau's identity
       f*/g* are the convex dual functions of g/f
    """

    @abc.abstractmethod
    def _prox_f(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_f^sigma"""
        raise NotImplementedError

    def prox(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_g^sigma using pre/post composition rules"""
        return self._prox_f(x - self.shift,
                            sigma=self.scale * sigma) + self.shift

    def prox_convex_dual(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_g*^sigma using Moreau"""
        return x - sigma * self.prox(x / sigma, sigma=1. / sigma)


class ConvexFunctionalWithDualProx(Functional):
    """abstract class for functional g(x) =  scale * f(x - shift) 
       where prox_f*^sigma is known (analytically)

       we can calculate prox_g^sigma using Moreau's identity
       f*/g* are the convex dual functions of g/f
    """

    @abc.abstractmethod
    def _prox_convex_dual_f(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_f*^sigma"""
        raise NotImplementedError

    def prox_convex_dual(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_g*^sigma"""
        return self.scale * self._prox_convex_dual_f(
            (x - sigma * self.shift) / self.scale, sigma=sigma / self.scale)

    def prox(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_g^sigma using Moreau"""
        return x - sigma * self.prox_convex_dual(x / sigma, sigma=1. / sigma)


class SquaredL2Norm(SmoothFunctional, ConvexFunctionalWithPrimalProx):
    """squared L2 norm times 0.5"""

    def _call_f(self, x: npt.NDArray | cpt.NDArray) -> float:
        """f(x) = 0.5 * sum_x conj(x_i) * x_i"""
        return float(0.5 * (self.xp.conj(x) * x).sum().real)

    def _prox_f(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_f^sigma = x / (1 + sigma)"""
        return x / (1 + sigma)

    def _gradient_f(self,
                    x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """gradient f = x"""
        return x


class L2L1Norm(ConvexFunctionalWithDualProx):
    """sum of pointwise Eucliean norms (L2L1 norm)"""

    def _call_f(self, x: npt.NDArray | cpt.NDArray) -> float:
        """f(x) = sum_i SquaredL2Norm(x_i)"""
        if self.xp.isrealobj(x):
            res = self.xp.linalg.norm(x, axis=0).sum()
        else:
            res = self.xp.linalg.norm(
                x.real, axis=0).sum() + self.xp.linalg.norm(x.imag,
                                                            axis=0).sum()

        return float(res)

    def _prox_convex_dual_f(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_f*^sigma = projection on L2 balls"""
        if self.xp.isrealobj(x):
            gnorm = self.xp.linalg.norm(x, axis=0)
            r = x / self.xp.clip(gnorm, 1, None)
        else:
            r = self.xp.zeros_like(x)

            gnorm_real = self.xp.linalg.norm(x.real, axis=0)
            r.real = x.real / self.xp.clip(gnorm_real, 1, None)

            gnorm_imag = self.xp.linalg.norm(x.imag, axis=0)
            r.imag = x.imag / self.xp.clip(gnorm_imag, 1, None)

        return r


class L1Norm(ConvexFunctionalWithDualProx):
    """L1 norm - sum of absolute values"""

    def _call_f(self, x: npt.NDArray | cpt.NDArray) -> float:
        """f(x) = sum_i SquaredL2Norm(x_i)"""
        if self.xp.isrealobj(x):
            res = self.xp.abs(x).sum()
        else:
            res = self.xp.abs(x.real).sum() + self.xp.abs(x.imag).sum()

        return float(res)

    def _prox_convex_dual_f(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        if self.xp.isrealobj(x):
            r = self.xp.clip(x, -1, 1)
        else:
            r = self.xp.zeros_like(x)

            r.real = self.xp.clip(x.real, -1, 1)
            r.imag = self.xp.clip(x.imag, -1, 1)

        return r