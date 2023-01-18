import types
import numpy.typing as npt
import math
from scipy.optimize import fmin_cg

try:
    import cupy.typing as cpt
except ModuleNotFoundError:
    import numpy.typing as cpt

from .functionals import ConvexFunctionalWithProx, SmoothFunctional, SquaredL2Norm
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

    @x.setter
    def x(self, value) -> None:
        self._x = value

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


class PDHG_ALG12:
    """(accelerated) primal-dual hybrid gradient algorithm 2 for optimizing
       F(operator x) + G(x) with
       gradient G lipschitz"""

    def __init__(self,
                 operator: LinearOperator,
                 f_functional: ConvexFunctionalWithProx,
                 g_functional: ConvexFunctionalWithProx,
                 grad_g_lipschitz: float | None = None,
                 sigma: float | None = None,
                 tau: float | None = None) -> None:
        """
        Parameters
        ----------
        operator : operators.LinearOperator
            linear operator
        f_functional : functionals.ConvexFunctionalWithProx
            F functional
        g_functional : functionals.ConvexFunctionalWithProx
            the G functional
        grad_g_lipschitz : float | None
            Lipschitz constant of the gradient of G
            if None, PDHG ALG1 (no acceleration) i used
            if not None, PDHG ALG2 (with step size update) is used
        sigma : float | None
            primal step size, if None 1/operator_norm 
        tau : float | None
            dual step size, if None 1/operator_norm 
        """

        self._operator = operator
        self._f_functional = f_functional

        operator_norm = None

        if sigma is None:
            operator_norm = operator.norm()
            self._sigma = 0.99 / operator_norm
        else:
            self._sigma = sigma

        if tau is None:
            if operator_norm is None:
                operator_norm = operator.norm()
            self._tau = 0.99 / operator_norm
        else:
            self._tau = tau

        self._theta = 0.999

        self._g_functional = g_functional
        self._grad_g_lipschitz = grad_g_lipschitz

        self._x = self.xp.zeros(self._operator.input_shape,
                                dtype=self._operator.input_dtype)
        self._xbar = self.xp.zeros(self._operator.input_shape,
                                   dtype=self._operator.input_dtype)
        self._y = self.xp.zeros(self._operator.output_shape,
                                dtype=self._operator.output_dtype)

        self.setup()

    @property
    def operator(self) -> LinearOperator:
        return self._operator

    @property
    def f_functional(self) -> ConvexFunctionalWithProx:
        return self._f_functional

    @property
    def g_functional(self) -> ConvexFunctionalWithProx:
        return self._g_functional

    @property
    def grad_g_lipschitz(self) -> float | None:
        return self._grad_g_lipschitz

    @property
    def xp(self) -> types.ModuleType:
        return self._operator.xp

    @property
    def x(self) -> npt.NDArray | cpt.NDArray:
        return self._x

    @x.setter
    def x(self, value) -> None:
        self._x = value

    @property
    def xbar(self) -> npt.NDArray | cpt.NDArray:
        return self._xbar

    @xbar.setter
    def xbar(self, value) -> None:
        self._xbar = value

    @property
    def y(self) -> npt.NDArray | cpt.NDArray:
        return self._y

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
    def theta(self) -> float:
        return self._theta

    @theta.setter
    def theta(self, value) -> None:
        self._theta = value

    @property
    def epoch_counter(self) -> int:
        return self._epoch_counter

    @property
    def cost(self) -> list[float]:
        return self._cost

    def setup(self) -> None:
        self._x = self.xp.zeros(self._operator.input_shape,
                                dtype=self._operator.input_dtype)
        self._xbar = self.xp.zeros(self._operator.input_shape,
                                   dtype=self._operator.input_dtype)
        self._y = self.xp.zeros(self._operator.output_shape,
                                dtype=self._operator.output_dtype)

        self._epoch_counter = 0
        self._cost = []

    def update(self) -> None:
        # prior operator forward step
        self._y = self._y + self.sigma * self.operator.forward(self._xbar)
        self._y = self.f_functional.prox_convex_dual(self._y, sigma=self.sigma)

        x_plus = self._x - self.tau * self.operator.adjoint(self._y)
        x_plus = self.g_functional.prox(x_plus, self.tau)

        # update the step sizes
        if not self.grad_g_lipschitz is None:
            self.theta = 1 / math.sqrt(1 +
                                       2 * self.grad_g_lipschitz * self.tau)
            self.tau = self.theta * self.tau
            self.sigma = self.theta / self.sigma

        self._xbar = x_plus + self.theta * (x_plus - self._x)
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
                self._cost.append(
                    self.f_functional(self.operator.forward(self.x)) +
                    self.g_functional(self.x))


class ADMM:
    """ADMM to minimize f(x) + g(Kx) 
       where f(x) is the smooth data fidelity with known gradient
       K is the gradient operator
       and g(.) is a gradient norm (e.g. non-smooth L1L2norm -> Total Variation)
    """

    def __init__(self, data_operator: LinearOperator,
                 data_distance: SmoothFunctional,
                 prior_operator: LinearOperator,
                 prior_functional: ConvexFunctionalWithProx) -> None:

        self._data_operator = data_operator
        self._data_distance = data_distance
        self._prior_operator = prior_operator
        self._prior_functional = prior_functional
        self._rho = 1.

        self._max_num_cg_iterations = 100
        self._cg_kwargs = {}

        self._cost_data = []
        self._cost_prior = []

        ######################
        # initialize variables
        ######################

        self._x = self.xp.zeros(data_operator.input_shape,
                                dtype=data_operator.input_dtype)
        # since we will pass x to scipy's fmin_cg, we have flatten it and convert it
        # to pseudo complex
        self._x = data_operator.ravel_pseudo_complex(self._x)

        self._u = self.xp.zeros(prior_operator.output_shape,
                                dtype=prior_operator.output_dtype)
        self._z = self.xp.zeros(prior_operator.output_shape,
                                dtype=prior_operator.output_dtype)

        # the actual data fidelity as a function of the image x (not expected_data(x))
        # is the data_distance evaluated at (data_operator.forward(x))
        self._data_fidelity = lambda z: self._data_distance(
            self._data_operator.forward(
                self._data_operator.unravel_pseudo_complex(z)))

        # using the chain rule, we can calculate the gradient of the
        # data fidelity term with respect to the image x which is given by
        # data_operator.adjoint( data_distance.gradient( data_operator.forward(x) ) )
        self._data_fidelity_gradient = lambda z: self._data_operator.ravel_pseudo_complex(
            self._data_operator.adjoint(
                self._data_distance.gradient(
                    self._data_operator.forward(
                        self._data_operator.unravel_pseudo_complex(z)))))

        # since scipy's optimization algorithms (e.g. fmin_cg) can only handle "flat"
        # real input arrays, we use the "(un)ravel complex" methods of the linear
        # operator class to transform flattened real arrays into unflattened complex arrays

    @property
    def x(self) -> npt.NDArray | cpt.NDArray:
        return self._data_operator.unravel_pseudo_complex(self._x)

    @x.setter
    def x(self, value) -> None:
        self._x = self._data_operator.ravel_pseudo_complex(value)

    @property
    def u(self) -> npt.NDArray | cpt.NDArray:
        return self._u

    @property
    def z(self) -> npt.NDArray | cpt.NDArray:
        return self._z

    @property
    def rho(self) -> float:
        return self._rho

    @rho.setter
    def rho(self, value) -> None:
        self._rho = value

    @property
    def xp(self) -> types.ModuleType:
        return self._data_operator.xp

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
    def cg_kwargs(self) -> dict:
        return self._cg_kwargs

    @cg_kwargs.setter
    def cg_kwargs(self, value: dict) -> None:
        self._cg_kwargs = value

    def update(self) -> None:
        ################################################################################
        # 1st ADMM subproblem - x update = argmin_x f(x) + 0.5*rho*||Kx - (z - u) ||_2^2
        ################################################################################

        # we setup a new function (and its gradient) that is the sum of the data fidelity f(x)
        # and the added quadratic term

        extra_quadratic_norm = SquaredL2Norm(xp=self.xp)
        extra_quadratic_norm.shift = (self._z - self._u)
        # the scale here is only rho instead of rho/2 since the SquaredL2Norm is 1/2 ||.||_2^2
        extra_quadratic_norm.scale = self._rho

        extra_quadratic = lambda y: self._prior_functional(
            self._prior_operator.forward(
                self._prior_operator.unravel_pseudo_complex(y)))

        extra_quadratic_gradient = lambda y: self._prior_operator.ravel_pseudo_complex(
            self._prior_operator.adjoint(
                extra_quadratic_norm.gradient(
                    self._prior_operator.forward(
                        self._prior_operator.unravel_pseudo_complex(y)))))

        # combine the data fidelity and the extra quadratic into a signle function and gradient
        # that we can pass to scipy's fmin_cg

        loss = lambda y: self._data_fidelity(y) + extra_quadratic(y)
        loss_gradient = lambda y: self._data_fidelity_gradient(
            y) + extra_quadratic_gradient(y)

        x0 = self._x.copy()

        res = fmin_cg(loss,
                      x0,
                      fprime=loss_gradient,
                      maxiter=self._max_num_cg_iterations,
                      **self._cg_kwargs)

        self._x = res.copy()

        ################################################################################
        # 2nd ADMM subproblem - z update = prox_g_(1/rho) (Kx + u)
        ################################################################################

        self._z = self._prior_functional.prox(
            self._prior_operator.forward(self.x) + self._u, 1 / self._rho)

        ################################################################################
        # 3rd ADMM subproblem - u update = u + (Kx - z)
        ################################################################################

        self._u += (self._prior_operator.forward(self.x) - self._z)

    def run(self, num_iterations: int, calculate_cost=False) -> None:
        for _ in range(num_iterations):
            self.update()
            if calculate_cost:
                self._cost_data.append(self._data_fidelity(self._x))
                if self._prior_functional is not None:
                    self._cost_prior.append(
                        self._prior_functional(
                            self._prior_operator.forward(self.x)))
