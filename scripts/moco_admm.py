"""minimal 1D demo to check whether Fred's approgridimation for ADMM sub problem (2) is valid"""
import numpy as np
import abc
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from typing import Callable


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


class Identity1DTransform(SpatialTransform):

    def forward(self, grid: np.ndarray) -> np.ndarray:
        return grid

    def inverse(self, warped_grid: np.ndarray) -> np.ndarray:
        return warped_grid


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


class CostFunction:

    def __init__(self, signal: InterpolatedSignal, data: np.ndarray,
                 data_operator: np.ndarray, prior: Callable[[np.ndarray],
                                                            float]):
        self._signal = signal
        self._prior = prior
        self._data = data
        self._data_operator = data_operator

    def __call__(self, z: np.ndarray) -> np.ndarray:
        diff = (
            self._data_operator @ self._signal.forward_warp(z)) - self._data
        data_fidelity = 0.5 * (diff**2).sum()

        return data_fidelity + self._prior(z)


class CostFunctionSubProblem2:

    def __init__(self, signal: InterpolatedSignal, v: np.ndarray, rho: float,
                 prior: Callable[[np.ndarray], float]):
        self._signal = signal
        self._v = v
        self._v_inverse_warped = self._signal.inverse_warp(v)
        self._rho = rho
        self._prior = prior

    def __call__(self, z):
        diff = self._signal.forward_warp(z) - self._v
        data_fidelity = 0.5 * self._rho * (diff**2).sum()
        return data_fidelity + self._prior(z)

    def __call2__(self, z):
        diff = z - self._v_inverse_warped
        data_fidelity = 0.5 * self._rho * (diff**2).sum()
        return data_fidelity + self._prior(z)


class TVprior:

    def __init__(self, beta: float = 1):
        self._beta = beta

    def __call__(self, x: np.ndarray) -> float:
        return self._beta * np.abs(x[1:] - x[:-1]).mean()


class QUADprior:

    def __init__(self, beta: float = 1):
        self._beta = beta

    def __call__(self, x: np.ndarray) -> float:
        return self._beta * ((x[1:] - x[:-1])**2).mean()


#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------

rhos: list[float] = [1e-2, 1e-1, 1e0, 1e1, 1e2]
noise_level: float = 0.5
num_iter: int = 500

prior = QUADprior(beta=0.1)

np.random.seed(1)

true_image = gaussian_filter(np.pad(2 * np.random.rand(32) - 1, 6), 1.75)
true_image[0] = 0
true_image[-1] = 0

n = true_image.size

grid = np.arange(true_image.size).astype(float)

# setup the non-rigid transform
transform = QuadraticSpatial1DTransform(grid.max())
#transform = Identity1DTransform()

warped_grid = transform.forward(grid)
signal = InterpolatedSignal(grid, transform=transform)

deformed_true_image = signal.forward_warp(true_image)

# setup the data operator and generate the data
A = gaussian_filter(np.random.rand(2 * n, n), 1.)
A[A < np.percentile(A, 30)] = 0
d_noise_free = (A @ deformed_true_image)
d = d_noise_free + noise_level * np.random.rand(d_noise_free.size)

# setup the total cost function we want to minimize
total_cost_func = CostFunction(signal, d, A, prior)

cost_array = np.zeros((len(rhos), num_iter))
cost_array2 = np.zeros((len(rhos), num_iter))
recons = np.zeros((len(rhos), n))
recons2 = np.zeros((len(rhos), n))

for irho, rho in enumerate(rhos):
    # opertators we need for the analytic solution of subproblem 1
    d_back = A.T @ d
    admm_operator_1 = np.linalg.inv(A.T @ A + rho * np.eye(n))

    #------------------------------------------------------
    # ADMM using original subproblem 2

    # intitialze variables
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    for iteration in range(num_iter):
        print(f'{(iteration+1):03}/{num_iter:03}')
        #------------------------------------------------------
        # ADMM subproblem 1
        # argmin_x 0.5*||Ax - d||_2^2  + 0.5*rho*||x - (Sz^k - u^k)||_2^2
        # which has the analytic solution (A^TA + rho*I)^-1 (A^T d + rho(Sz^k - u^k))

        deformed_z = signal.forward_warp(z)
        x = admm_operator_1 @ (d_back + rho * (deformed_z - u))

        #------------------------------------------------------
        # ADMM subproblem 2
        # argmin g(z) + 0.5*rho*||Sz -v||_2^2 with v = x^(k+1) + u^k

        cost = CostFunctionSubProblem2(signal, (x + u), rho=rho, prior=prior)
        res1 = minimize(cost.__call__, z)
        z = res1.x.copy()

        #------------------------------------------------------
        # update u
        deformed_z = signal.forward_warp(z)
        u += (x - signal.forward_warp(z))

        #------------------------------------------------------
        # calculate the value of the total cost function
        cost_array[irho, iteration] = total_cost_func(z)

    recons[irho, ...] = z.copy()

    #------------------------------------------------------
    # ADMM using modified subproblem 2

    # intitialze variables
    x2 = np.zeros(n)
    z2 = np.zeros(n)
    u2 = np.zeros(n)

    for iteration in range(num_iter):
        print(f'{(iteration+1):03}/{num_iter:03}')
        #------------------------------------------------------
        # ADMM subproblem 1
        # argmin_x 0.5*||Ax - d||_2^2  + 0.5*rho*||x - (Sz^k - u^k)||_2^2
        # which has the analytic solution (A^TA + rho*I)^-1 (A^T d + rho(Sz^k - u^k))

        deformed_z2 = signal.forward_warp(z2)
        x2 = admm_operator_1 @ (d_back + rho * (deformed_z2 - u2))

        #------------------------------------------------------
        # ADMM subproblem 2
        # argmin g(z) + 0.5*rho*||Sz -v||_2^2 with v = x^(k+1) + u^k

        cost = CostFunctionSubProblem2(signal, (x2 + u2), rho=rho, prior=prior)
        res2 = minimize(cost.__call2__, z2)
        z2 = res2.x.copy()

        #------------------------------------------------------
        # update u
        deformed_z2 = signal.forward_warp(z2)
        u2 += (x2 - signal.forward_warp(z2))

        #------------------------------------------------------
        # calculate the value of the total cost function
        cost_array2[irho, iteration] = total_cost_func(z2)

    recons2[irho, ...] = z2.copy()

#--------------------------------------------------------------------------------
# calculate a reference solution by brute force minimization
ref_solution = minimize(total_cost_func.__call__, z)
z_ref = ref_solution.x.copy()

#--------------------------------------------------------------------------------
# plots

fig, ax = plt.subplots(2,
                       len(rhos),
                       figsize=(4 * len(rhos), 2 * 4),
                       sharey='row')

for irho, rho in enumerate(rhos):
    #ax[0, irho].plot(deformed_true_image, '.-', label='true deformed')
    ax[0, irho].plot(true_image, '.-', label='true', color='black')
    ax[0, irho].plot(recons[irho, ...], '.-', label='recon1', color='blue')
    ax[0, irho].plot(recons2[irho, ...], '.-', label='recon2', color='red')
    ax[0, irho].legend()
    ax[1, irho].loglog(np.arange(1, num_iter + 1),
                       cost_array[irho, ...],
                       label='cost1',
                       color='blue')
    ax[1, irho].loglog(np.arange(1, num_iter + 1),
                       cost_array2[irho, ...],
                       label='cost2',
                       color='red')
    ax[1, irho].axhline(ref_solution.fun, color='black')
    ax[1, irho].legend()

    ax[0, irho].set_title(f'rho {rho:.2E}')

for axx in ax.ravel():
    axx.grid(ls=':')

fig.tight_layout()
fig.show()
