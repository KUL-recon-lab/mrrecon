"""minimal 1D demo to check whether Fred's approgridimation for ADMM sub problem (2) is valid"""
import numpy as np
import abc
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


class SpatialTransform(abc.ABC):

    @abc.abstractmethod
    def forward(self, grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, warped_grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class QuadraticSpatial1DTransform(SpatialTransform):
    """1D nonlinear quadratic transformation"""

    def __init__(self, a: float):
        self._a = a

    @property
    def a(self) -> float:
        return self._a

    def forward(self, grid: np.ndarray) -> np.ndarray:
        return (-(grid - self.a)**2 + self.a**2) / self.a

    def inverse(self, warped_grid: np.ndarray) -> np.ndarray:
        return -np.sqrt(self.a**2 - self.a * warped_grid) + self.a


def interp_kernel(grid: float, center: float = 0, amplitude: float = 1.):
    """linear (triangle) interpolation kernel"""
    y = np.clip((1 - np.abs(grid - center)), 0, None)
    return (amplitude * y)


class InterpolatedSignal:

    def __init__(self, grid: np.ndarray, transform: SpatialTransform) -> None:
        self._grid = grid
        self._n = grid.shape[0]
        self._forward_warped_grid = transform.forward(self._grid)
        self._inverse_warped_grid = transform.inverse(self._grid)

    def __call__(self, amplitudes: np.ndarray) -> np.ndarray:
        y = np.zeros(self._n)
        for i, amp in enumerate(amplitudes):
            y += interp_kernel(self._grid, center=i, amplitude=amp)
        return y

    def forward_warp(self, amplitudes: np.ndarray) -> np.ndarray:
        y = np.zeros(self._n)
        for i, amp in enumerate(amplitudes):
            y += interp_kernel(self._forward_warped_grid,
                               center=i,
                               amplitude=amp)
        return y

    def inverse_warp(self, amplitudes: np.ndarray) -> np.ndarray:
        y = np.zeros(self._n)
        for i, amp in enumerate(amplitudes):
            y += interp_kernel(self._inverse_warped_grid,
                               center=i,
                               amplitude=amp)
        return y


#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------

np.random.seed(1)

amps = gaussian_filter(np.random.rand(64), 2.5)

grid = np.arange(amps.size).astype(float)
grid_highres = np.linspace(0, grid.max(), 10000)

transform = QuadraticSpatial1DTransform(grid.max())
warped_grid = transform.forward(grid)
warped_grid_highres = transform.forward(grid_highres)

signal = InterpolatedSignal(grid, transform=transform)
signal_highres = InterpolatedSignal(grid_highres, transform=transform)

#--------------------------------------------------------------------------------
amps2 = signal.forward_warp(amps)
amps3 = signal.inverse_warp(amps2)

#--------------------------------------------------------------------------------
# plots

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(grid, warped_grid)
ax[0].plot(grid, grid, 'k:')
ax[1].plot(grid_highres, signal_highres(amps), 'b', lw=0.5)
ax[1].plot(grid_highres, signal_highres.forward_warp(amps), 'r', lw=0.5)
ax[1].plot(grid, signal(amps), 'bo')
ax[1].plot(grid, signal.forward_warp(amps), 'r.')

for axx in ax.ravel():
    axx.grid(ls=':')
fig.tight_layout()
fig.show()

#fig2, ax2 = plt.subplots()
#ax2.plot(amps, 'o-')
#ax2.plot(amps2, '.')
#ax2.plot(amps3, '.')
#fig2.show()