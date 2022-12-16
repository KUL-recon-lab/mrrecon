"""generic linear operators related to imaging problems"""
import abc
import types
import numpy as np
import numpy.typing as npt

try:
    import cupy.typing as cpt
except ModuleNotFoundError:
    import numpy.typing as cpt


class LinearOperator(abc.ABC):

    def __init__(self,
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 xp: types.ModuleType = np,
                 input_dtype: type = float,
                 output_dtype: type = float,
                 pre_scale: float | npt.NDArray | cpt.NDArray = 1,
                 post_scale: float | npt.NDArray | cpt.NDArray = 1) -> None:
        """Linear operator abstract base class that maps real or complex array x to y
            diag(post_scale) A (diag(pre_scale) x)

        Parameters
        ----------
        input_shape : tuple
            shape of x array
        output_shape : tuple
            shape of y array
        xp : types.ModuleType | None
            module indicating whether to store all LOR endpoints as numpy as cupy array
            by default numpy
        input_dtype: type
            data type of the input array
            by default float
        output_dtype: type
            data type of the input array
            by default float
        pre_scale : float | npt.NDArray | cpt.NDArray
            scalar / pointwise scaling before applying the operator
        post_scale : float | npt.NDArray | cpt.NDArray
            scalar / pointwise scaling after applying the operator
        """

        self._input_shape = input_shape
        self._output_shape = output_shape

        self._xp = xp

        self._input_dtype = input_dtype
        self._output_dtype = output_dtype

        self._pre_scale = pre_scale
        self._post_scale = post_scale

    @property
    def input_dtype(self) -> type:
        """the data type of the input array"""
        return self._input_dtype

    @input_dtype.setter
    def input_dtype(self, value) -> None:
        self._input_dtype = value

    @property
    def output_dtype(self) -> type:
        """the data type of the output array"""
        return self._output_dtype

    @output_dtype.setter
    def output_dtype(self, value) -> None:
        self._output_dtype = value

    @property
    def input_shape(self) -> tuple[int, ...]:
        """shape of the input array"""
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value) -> None:
        self._input_shape = value

    @property
    def output_shape(self) -> tuple[int, ...]:
        """shape of the output array"""
        return self._output_shape

    @output_shape.setter
    def output_shape(self, value) -> None:
        self._output_shape = value

    @property
    def xp(self) -> types.ModuleType:
        """module indicating whether the LOR endpoints are stored as numpy or cupy array"""
        return self._xp

    @property
    def pre_scale(self) -> float | npt.NDArray | cpt.NDArray:
        return self._pre_scale

    @pre_scale.setter
    def pre_scale(self, value) -> None:
        self._pre_scale = value

    @property
    def post_scale(self) -> float | npt.NDArray | cpt.NDArray:
        return self._post_scale

    @post_scale.setter
    def post_scale(self, value) -> None:
        self._post_scale = value

    @abc.abstractmethod
    def _forward(self,
                 x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """forward step

        Parameters
        ----------
        x : npt.NDArray | cpt.NDArray
            x array

        Returns
        -------
        npt.NDArray | cpt.NDArray
            the linear operator applied to x
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _adjoint(self,
                 y: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """adjoint of forward step

        Parameters
        ----------
        y : npt.NDArray | cpt.NDArray
            y array

        Returns
        -------
        npt.NDArray | cpt.NDArray
            the adjoint of the linear operator applied to y
        """
        raise NotImplementedError()

    def forward(self,
                x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:

        return self.post_scale * self._forward(self.pre_scale * x)

    def adjoint(self,
                y: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:

        return self.xp.conj(self.pre_scale) * self._adjoint(
            self.xp.conj(self.post_scale) * y)

    def adjointness_test(self, verbose=False, **kwargs) -> None:
        """test if adjoint is really the adjoint of forward

        Parameters
        ----------
        verbose : bool, optional
            print verbose output
        **kwargs : key word arguments
            passed to xp.isclose()
        """
        x = self.xp.random.rand(*self.input_shape).astype(self.input_dtype)
        y = self.xp.random.rand(*self.output_shape).astype(self.output_dtype)

        if self.xp.iscomplexobj(x):
            x += 1j * self.xp.random.rand(*self.input_shape).astype(
                self.input_dtype)

        if self.xp.iscomplexobj(y):
            y += 1j * self.xp.random.rand(*self.output_shape).astype(
                self.output_dtype)

        x_fwd = self.forward(x)
        y_back = self.adjoint(y)

        a = (self.xp.conj(y) * x_fwd).sum()
        b = (self.xp.conj(y_back) * x).sum()

        if verbose:
            print(f'<y, A x>   {a}')
            print(f'<A^T y, x> {b}')

        assert (self.xp.isclose(a, b, **kwargs))

    def norm(self, num_iter=20) -> float:
        """estimate norm of operator via power iterations

        Parameters
        ----------
        num_iter : int, optional
            number of iterations, by default 20

        Returns
        -------
        float
            the estimated norm
        """

        x = self.xp.random.rand(*self._input_shape).astype(self.input_dtype)

        if self.xp.iscomplexobj(x):
            x += 1j * self.xp.random.rand(*self.input_shape).astype(
                self.input_dtype)

        for i in range(num_iter):
            x = self.adjoint(self.forward(x))
            n = self.xp.linalg.norm(x.ravel())
            x /= n

        return float(self.xp.sqrt(n))

    def unravel_pseudo_complex(
            self, x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """unravel a real flattened pseudo-complex (pair of 2 reals) array into a complex array

        Parameters
        ----------
        x : npt.NDArray | cpt.NDArray
            real flattened array with size 2*prod(input_shape)

        Returns
        -------
        npt.NDArray | cpt.NDArray
            unraveled complex array
        """

        x = x.reshape(self.input_shape + (2, ))

        if x.dtype == self.xp.float64:
            return self.xp.squeeze(x.view(dtype=self.xp.complex128), axis=-1)
        elif x.dtype == self.xp.float32:
            return self.xp.squeeze(x.view(dtype=self.xp.complex64), axis=-1)
        elif x.dtype == self.xp.float128:
            return self.xp.squeeze(x.view(dtype=self.xp.complex256), axis=-1)
        else:
            raise ValueError(
                'Input must have dtyoe float32, float64 or float128')

    def ravel_pseudo_complex(
            self, x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """ravel a complex array into a flattened pseudo-complex (pair of 2 reals) array

        Parameters
        ----------
        x : npt.NDArray | cpt.NDArray
            complex array of shape input_shape
        Returns
        -------
        npt.NDArray | cpt.NDArray
            raveled pseudo complex array of shape 2*prod(input_shape)
        """
        return self.xp.stack([x.real, x.imag], axis=-1).ravel()


class GradientOperator(LinearOperator):
    """finite difference gradient operator for real or complex arrays"""

    def __init__(self,
                 input_shape: tuple[int, ...],
                 xp: types.ModuleType = np,
                 dtype: type = float) -> None:
        """_summary_

        Parameters
        ----------
        input_shape : tuple[int, ...]
            the input array shape
        xp : types.ModuleType
            the array module (numpy or cupy)
            by default numpy
        dtype : type, optional,
            data type of the input array, 
            by default float
        """

        output_shape = (len(input_shape), ) + input_shape
        super().__init__(input_shape,
                         output_shape,
                         xp=xp,
                         input_dtype=dtype,
                         output_dtype=dtype)

    def _forward(self, x):
        g = self.xp.zeros(self.output_shape, dtype=self.output_dtype)
        for i in range(x.ndim):
            g[i, ...] = self.xp.diff(x,
                                     axis=i,
                                     append=self.xp.take(x, [-1], i))

        return g

    def _adjoint(self, y):
        d = self.xp.zeros(self.input_shape, dtype=self.input_dtype)

        for i in range(y.shape[0]):
            tmp = y[i, ...]
            sl = [slice(None)] * y.shape[0]
            sl[i] = slice(-1, None)
            tmp[tuple(sl)] = 0
            d -= self.xp.diff(tmp, axis=i, prepend=0)

        return d