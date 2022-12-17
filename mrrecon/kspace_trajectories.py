import types
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

try:
    import cupy.typing as cpt
except ModuleNotFoundError:
    import numpy.typing as cpt


def radial_2d_golden_angle(
        num_spokes: int,
        num_samples_per_spoke: int,
        kmax: float = np.pi,
        mode: str = 'half-spoke',
        golden_angle: None | float = None,
        xp: types.ModuleType = np) -> npt.NDArray | cpt.NDArray:

    if mode == 'half-spoke':
        if golden_angle is None:
            golden_angle = xp.pi * 137.51 / 180
        k1d = xp.linspace(0, kmax, num_samples_per_spoke, endpoint=True)
    elif mode == 'full-spoke':
        if golden_angle is None:
            golden_angle = xp.pi * 111.25 / 180
        k1d = xp.linspace(-kmax, kmax, num_samples_per_spoke, endpoint=False)
    else:
        raise ValueError

    spoke_angles = (xp.arange(num_spokes) * golden_angle) % (2 * xp.pi)

    k = xp.zeros((num_spokes * num_samples_per_spoke, 2))

    for i, angle in enumerate(spoke_angles):
        k[(i * num_samples_per_spoke):((i + 1) * num_samples_per_spoke),
          0] = xp.cos(angle) * k1d
        k[(i * num_samples_per_spoke):((i + 1) * num_samples_per_spoke),
          1] = xp.sin(angle) * k1d

    return k


def stack_of_2d_golden_angle(num_stacks: int,
                             kzmax: float = np.pi,
                             xp: types.ModuleType = np,
                             **kwargs) -> npt.NDArray | cpt.NDArray:

    star_kspace_sample_points = radial_2d_golden_angle(xp=xp, **kwargs)
    num_star_samples = star_kspace_sample_points.shape[0]

    kz1d = xp.linspace(-kzmax, kzmax, num_stacks, endpoint=False)

    k = xp.zeros((num_stacks * num_star_samples, 3))

    for i, kz in enumerate(kz1d):
        start = i * num_star_samples
        end = (i + 1) * num_star_samples

        k[start:end, 0] = kz
        k[start:end, 1:] = star_kspace_sample_points

    return k


class TPITrajectory:

    def __init__(self, fname: str, kmax: float = 1.) -> None:
        self._fname = fname
        self._kmax = kmax

        tmp = np.loadtxt(fname)
        self._t_ms = tmp[:, 1]
        self._k = tmp[:, 0] * self._kmax / tmp[:, 0].max()

        self._t_of_k = interp1d(self._k, self._t_ms)

    def t_of_k(self, k: npt.NDArray, factor: float = 1.) -> npt.NDArray:
        return factor * self._t_of_k(np.abs(k))
