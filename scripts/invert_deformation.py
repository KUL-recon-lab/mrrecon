"""mini example to test fix point iteration scheme to invert deformation fields

f: function that maps from domain A to B (forwad deformation)
u(x) = f(x) - x (forward displacement field)

g: inverse of f
v(x) = g(x) - x (inverse displacement field)

since g is the inverse of f, we have:
x = f(g(x)) = g(x) + u(g(x)) = x + v(x) + u(x + v(x))

-> v(x) = -u(x + v(x))

iterative scheme:

v_0 = 0
v_n = -u(x + v_{n-1})
"""

import numpy as np
import matplotlib.pyplot as plt
import abc


class ForwardDisplacement(abc.ABC):

    @abc.abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fixpoint_inverse(self, x: np.ndarray, n: int) -> np.ndarray:
        if n == 0:
            return np.zeros_like(x)
        else:
            return -self.__call__(x + self.fixpoint_inverse(x, n - 1))


class LinearDisplacement(ForwardDisplacement):

    def __init__(self, a: float, b: float) -> None:
        self._a = a
        self._b = b

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._a + self._b * x

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape, self._b)


class QuadraticDisplacement(ForwardDisplacement):

    def __init__(self, a: float, b: float, c: float) -> None:
        self._a = a
        self._b = b
        self._c = c

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._a + self._b * x + self._c * (x**2)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape, self._b + 2 * self._c * x)


class TestDisplacement(ForwardDisplacement):

    def __init__(self, a: float = 1, b: float = 1, c: float = 0) -> None:
        self._a = a
        self._b = b
        self._c = c

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._b * ((1 / (1 + self._a * x)) - 1) + self._c

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return self._b * (-self._a / (1 + x)**2)


#------------------------------------------------------------------------------

if __name__ == '__main__':

    x = np.linspace(0, 2, 100)

    #u = LinearDisplacement(0, 0.8)
    #u = QuadraticDisplacement(0, 0.3, 0.2)
    u = TestDisplacement(a=1., b=1., c=-0.5)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(x, u(x))
    ax[1].plot(x, u(x) + x)
    ax[2].plot(x, x + u(x))  # plot(x,f)
    for n in [1, 3, 10, 100, 1000]:
        ax[2].plot(u.fixpoint_inverse(x, n) + x, x)  # plot(g,x)
    fig.tight_layout()
    fig.show()