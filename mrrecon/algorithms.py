import types
import numpy.typing as npt

try:
    import cupy.typing as cpt
except ModuleNotFoundError:
    import numpy.typing as cpt

from .functionals import ConvexFunctionalWithProx
from .linearoperators import LinearOperator


class PDHG:
    """generic primal-dual hybrid gradient algorithm (Chambolle-Pock) for optimizing
       data_distance(data_operator x) + beta*(prior_functional(prior_operator x)) + g_functional(x)"""

    def __init__(self,
                 data_operator: LinearOperator,
                 data_distance: ConvexFunctionalWithProx,
                 sigma: float,
                 tau: float,
                 theta: float = 0.999,
                 prior_operator: LinearOperator | None = None,
                 prior_functional: ConvexFunctionalWithProx | None = None,
                 g_functional: ConvexFunctionalWithProx | None = None) -> None:
        """
        Parameters
        ----------
        data_operator : operators.LinearOperator
            operator mapping current image to expected data
        data_distance : functionals.ConvexFunctionalWithProx
            norm applied to (expected data - data)
        prior_operator : operators.LinearOperator
            prior operator
        prior_functional : functionals.ConvexFunctionalWithProx
            prior norm
        sigma : float
            primal step size 
        tau : float
            dual step size 
        theta : float, optional
            theta parameter, by default 0.999
        g_functional : None | functionals.ConvexFunctionalWithProx
            the G functional
        """

        self._data_operator = data_operator
        self._data_distance = data_distance

        self._prior_operator = prior_operator
        self._prior_functional = prior_functional

        self._sigma = sigma
        self._tau = tau
        self._theta = theta

        self._g_functional = g_functional

        self._x = self.xp.zeros(self._data_operator.input_shape,
                                dtype=self.data_operator.input_dtype)
        self._xbar = self.xp.zeros(self._data_operator.input_shape,
                                   dtype=self.data_operator.input_dtype)
        self._y_data = self.xp.zeros(self._data_operator.output_shape,
                                     dtype=self.data_operator.output_dtype)
        if self._prior_operator is not None:
            self._y_prior = self.xp.zeros(
                self._prior_operator.output_shape,
                dtype=self.prior_operator.output_dtype)

        self.setup()

    @property
    def data_operator(self) -> LinearOperator:
        return self._data_operator

    @property
    def prior_operator(self) -> LinearOperator | None:
        return self._prior_operator

    @property
    def data_distance(self) -> ConvexFunctionalWithProx:
        return self._data_distance

    @property
    def prior_functional(self) -> ConvexFunctionalWithProx | None:
        return self._prior_functional

    @property
    def g_functional(self) -> ConvexFunctionalWithProx | None:
        return self._g_functional

    @property
    def xp(self) -> types.ModuleType:
        return self._data_operator.xp

    @property
    def x(self) -> npt.NDArray | cpt.NDArray:
        return self._x

    @property
    def y_data(self) -> npt.NDArray | cpt.NDArray:
        return self._y_data

    @property
    def y_prior(self) -> npt.NDArray | cpt.NDArray:
        return self._y_prior

    @property
    def cost_data(self) -> npt.NDArray | cpt.NDArray:
        return self.xp.array(self._cost_data)

    @property
    def cost_prior(self) -> npt.NDArray | cpt.NDArray:
        return self.xp.array(self._cost_prior)

    @property
    def cost(self) -> npt.NDArray | cpt.NDArray:
        return self.cost_data + self.cost_prior

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, value) -> None:
        self._sigma = value

    @property
    def tau(self) -> float:
        return self._tau

    @tau.setter
    def tau(self, value) -> None:
        self._tau = value

    @property
    def epoch_counter(self) -> int:
        return self._epoch_counter

    def setup(self) -> None:
        self._x = self.xp.zeros(self.data_operator.input_shape,
                                dtype=self.data_operator.input_dtype)
        self._xbar = self.xp.zeros(self.data_operator.input_shape,
                                   dtype=self.data_operator.input_dtype)
        self._y_data = self.xp.zeros(self.data_operator.output_shape,
                                     dtype=self.data_operator.output_dtype)
        if self._prior_operator is not None:
            self._y_prior = self.xp.zeros(
                self.prior_operator.output_shape,
                dtype=self.prior_operator.output_dtype)

        self._epoch_counter = 0
        self._cost_data = []
        self._cost_prior = []

    def update(self) -> None:
        # data forward step
        self._y_data = self._y_data + self.sigma * self.data_operator.forward(
            self._xbar)

        # prox of data fidelity
        self._y_data = self.data_distance.prox_convex_dual(self._y_data,
                                                           sigma=self.sigma)

        if self.prior_functional is not None:
            # prior operator forward step
            self._y_prior = self._y_prior + self.sigma * self.prior_operator.forward(
                self._xbar)
            # prox of prior norm
            self._y_prior = self.prior_functional.prox_convex_dual(
                self._y_prior, sigma=self.sigma)

        x_plus = self._x - self.tau * self.data_operator.adjoint(self._y_data)

        if self.prior_functional is not None:
            x_plus -= self.tau * self.prior_operator.adjoint(self._y_prior)

        if self._g_functional is not None:
            x_plus = self.g_functional.prox(x_plus, self.tau)

        self._xbar = x_plus + self._theta * (x_plus - self._x)

        self._x = x_plus.copy()

        self._epoch_counter += 1

    def run(self,
            num_iterations: int,
            calculate_cost: bool = False,
            verbose: bool = True) -> None:

        for _ in range(num_iterations):
            self.update()
            if verbose:
                print(f'iteration {self.epoch_counter}')
            if calculate_cost:
                self._cost_data.append(
                    self._data_distance(self._data_operator.forward(self._x)))
                if self._prior_functional is not None:
                    self._cost_prior.append(
                        self._prior_functional(
                            self._prior_operator.forward(self._x)))
