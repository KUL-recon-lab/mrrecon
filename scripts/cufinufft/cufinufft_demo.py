import numpy as np 
 
import pycuda.autoinit # NOQA:401 
import pycuda.gpuarray as gpuarray 
 
from cufinufft import cufinufft 
 
import utils


from time import time
num_rep = 1

dtype = np.float32
shape = (64, 64, 64)
M = int(1e3)
tol = 1e-6
complex_dtype = utils._complex_dtype(dtype)

# sign in exp.
isign = -1

# setup M random real 3-d kspace points within [-pi, pi)^3
k = utils.gen_nu_pts(M).astype(dtype)
k[:,0] = 0
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

ind = 0

# execute type2 NUFFT (forward step going from regular image to non-unif. kspace)
t0 = time()
fk_gpu = gpuarray.to_gpu(fk)
c_gpu = gpuarray.GPUArray(shape=(M,), dtype=complex_dtype)

plan2.execute(c_gpu, fk_gpu)
res2 = c_gpu.get()
t1 = time()

res2_direct = utils.direct_type2(fk, k[:, ind])

# execute type1 NUFFT which should be the adjoint of the forwad step
t2 = time()
c_gpu = gpuarray.to_gpu(c)
fk_gpu = gpuarray.GPUArray(shape, dtype=complex_dtype)
plan1.execute(c_gpu, fk_gpu)
res1 = fk_gpu.get()
t3 = time()

print(f'<Ax,y>   : {(res2*np.conj(c)).sum():.3f}')
print(f'<x,A^H y>: {(fk*np.conj(res1)).sum():.3f}')
print(f't fwd: {(t1-t0):.5f}s')
print(f't adj: {(t3-t2):.5f}s')
print(f'type 2 rel err: {np.abs(res2_direct - res2[ind]) / np.abs(res2_direct)}')
print()


import sigpy

coords = k.T.copy()
for i in range(3):
    coords[:,i] *= (0.5*shape[i]/np.pi)

A = sigpy.linop.NUFFT(shape, k.T)

res2_sigpy = A(fk)

print(np.abs(res2[:5]) / np.abs(res2_sigpy[:5]))
