import numpy as np 
 
import pycuda.autoinit # NOQA:401 
import pycuda.gpuarray as gpuarray 
 
from cufinufft import cufinufft 
 
import utils


from time import time
num_rep = 3

dtype = np.float32
shape = (256, 256, 256)
M = int(1e6)
tol = 1e-3
complex_dtype = utils._complex_dtype(dtype)

# sign in exp.
isign = 1

# setup M random real 3-d kspace points within [-pi, pi)^3
k = utils.gen_nu_pts(M).astype(dtype)
# setup a random complex image
fk = utils.gen_uniform_data(shape).astype(complex_dtype)
# setup a random complex k-space data vector
c = utils.gen_nonuniform_data(M).astype(complex_dtype)

# send the k-space vector to GPU
k_gpu = gpuarray.to_gpu(k)


# type 2 NUFFT going from "uniform" image to "non-uniform" k-space points
plan2 = cufinufft(2, shape, eps=tol, dtype=dtype, isign = -isign)
plan2.set_pts(k_gpu[0], k_gpu[1], k_gpu[2])

# type 1 NUFFT going from "non-uniform" k-space points to "uniform" image 
plan1 = cufinufft(1, shape, eps=tol, dtype=dtype, isign = isign)
plan1.set_pts(k_gpu[0], k_gpu[1], k_gpu[2])

for i in range(num_rep):
    # execute type2 NUFFT (forward step going from regular image to non-unif. kspace)
    t0 = time()
    fk_gpu = gpuarray.to_gpu(fk)
    c_gpu = gpuarray.GPUArray(shape=(M,), dtype=complex_dtype)

    plan2.execute(c_gpu, fk_gpu)
    res2 = c_gpu.get()
    t1 = time()

    # execute type1 NUFFT which should be the adjoint of the forwad step
    t2 = time()
    c_gpu = gpuarray.to_gpu(c)
    fk_gpu = gpuarray.GPUArray(shape, dtype=complex_dtype)
    plan1.execute(c_gpu, fk_gpu)
    res1 = fk_gpu.get()
    t3 = time()

    print((res2*np.conj(c)).sum())
    print((fk*np.conj(res1)).sum())
    print(t1-t0)
    print(t3-t2)
    print()

