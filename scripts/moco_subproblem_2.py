"""minimal 1D demo to check whether Fred's approgridimation for ADMM sub problem (2) is valid"""
import numpy as np
import abc
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize


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


class CostFunctionSubProblem2:

    def __init__(self, interpolatad_signal: InterpolatedSignal,
                 v_amps: np.ndarray, rho: float):
        self._interpolatad_signal = interpolatad_signal
        self._v_amps = v_amps
        self._v_amps_inverse_warped = self._interpolatad_signal.inverse_warp(
            v_amps)
        self._rho = rho

    @staticmethod
    def prior(x):
        return np.abs(x[1:] - x[:-1]).mean()

    def __call__(self, z_amps):
        diff = self._interpolatad_signal.forward_warp(z_amps) - self._v_amps
        data_fidelity = 0.5 * self._rho * (diff**2).sum()
        return data_fidelity + self.prior(z_amps)

    def __call2__(self, z_amps):
        diff = z_amps - self._v_amps_inverse_warped
        data_fidelity = 0.5 * self._rho * (diff**2).sum()
        return data_fidelity + self.prior(z_amps)


#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------

rhos = [3e-2, 1e-1, 3e-1, 1e0, 3e0]

np.random.seed(1)

amps = gaussian_filter(np.pad(2 * np.random.rand(32) - 1, 6), 1.75)
amps[0] = 0
amps[-1] = 0

grid = np.arange(amps.size).astype(float)
grid_highres = np.linspace(0, grid.max(), 10000)

transform = QuadraticSpatial1DTransform(grid.max())
warped_grid = transform.forward(grid)
warped_grid_highres = transform.forward(grid_highres)

signal = InterpolatedSignal(grid, transform=transform)
signal_highres = InterpolatedSignal(grid_highres, transform=transform)

#--------------------------------------------------------------------------------
amps_fwd = signal.forward_warp(amps)
amps_fwd_inv = signal.inverse_warp(amps_fwd)

#--------------------------------------------------------------------------------
# solve the two different minimization problems
z0 = np.ones_like(amps)

z1 = np.zeros((len(rhos), amps.shape[0]))
z2 = np.zeros((len(rhos), amps.shape[0]))

for i, rho in enumerate(rhos):
    print(i, rho)
    cost = CostFunctionSubProblem2(signal, amps_fwd, rho=rho)
    res1 = minimize(cost.__call__, z0)
    res2 = minimize(cost.__call2__, z0)
    z1[i, :] = res1.x
    z2[i, :] = res2.x

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

fig2, ax2 = plt.subplots()
ax2.plot(amps, 'o-')
ax2.plot(amps_fwd, '.')
ax2.plot(amps_fwd_inv, '.')
fig2.show()

fig3, ax3 = plt.subplots(1,
                         len(rhos),
                         figsize=(len(rhos) * 3, 3),
                         sharex=True,
                         sharey=True)

for i, rho in enumerate(rhos):
    ax3[i].plot(z1[i, :])
    ax3[i].plot(z2[i, :])
    ax3[i].set_title(f'rho {rho:.2E}')
for axx in ax3.ravel():
    axx.grid(ls=':')
fig3.tight_layout()
fig3.show()
