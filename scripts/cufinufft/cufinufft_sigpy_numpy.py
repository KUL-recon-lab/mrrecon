""" compare cufinufft, sigpy nufft and numpy's fft
    this demo shows:
    - how to convert kspace coordinates between cufinufft, sigpy, numpy
    - how to calculate the phase factor missing in numpy's fft defnition
    - differences in the cufinufft and sigpy nufft
"""

import numpy as np 
 
import pycuda.autoinit # NOQA:401 
import pycuda.gpuarray as gpuarray 
 
from cufinufft import cufinufft 
 
import utils

from time import time
num_rep = 1

dtype = np.float32
n = 4 # should be even
shape = (n,n,n)
M = np.prod(shape)
tol = 1e-6
complex_dtype = utils._complex_dtype(dtype)

# sign in exp.
isign = 1

# setup M random real 3-d kspace points within [-pi, pi)^3
tmp = 2*np.pi * np.fft.fftfreq(n).astype(dtype)
k0, k1, k2 = np.meshgrid(tmp, tmp, tmp, indexing = 'ij')
k = np.array([k0,k1,k2]).reshape(3,-1)

# setup a random complex image
fk = utils.gen_uniform_data(shape).astype(complex_dtype)
# setup a random complex k-space data vector
c = utils.gen_nonuniform_data(M).astype(complex_dtype)

# type 2 NUFFT going from "uniform" image to "non-uniform" k-space points
plan2 = cufinufft(2, shape, eps=tol, dtype=dtype, isign = -isign)
plan2.set_pts(gpuarray.to_gpu(k[0,...]), gpuarray.to_gpu(k[1,...]), gpuarray.to_gpu(k[2,...]))

# type 1 NUFFT going from "non-uniform" k-space points to "uniform" image 
plan1 = cufinufft(1, shape, eps=tol, dtype=dtype, isign = isign)
plan1.set_pts(gpuarray.to_gpu(k[0,...]), gpuarray.to_gpu(k[1,...]), gpuarray.to_gpu(k[2,...]))

ind = 0

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

print(f'<Ax,y>   : {(res2*np.conj(c)).sum():.3f}')
print(f'<x,A^H y>: {(fk*np.conj(res1)).sum():.3f}')
print(f't fwd: {(t1-t0):.5f}s')
print(f't adj: {(t3-t2):.5f}s')
print()

#----------------------------------------------------------------------

import sigpy

# sigpy expects the k-space coordiantes in a transposed way
# and ranging from -N/2 ... (N/2 - 1)
coords = k.T.copy()
for i in range(len(shape)):
    coords[:,i] *= (0.5*shape[i]/np.pi)

A = sigpy.linop.NUFFT(shape, coords, oversamp = 2.)

res2_sigpy = A(fk) * np.sqrt(np.prod(shape)) 

#----------------------------------------------------------------------

# calculate numpy fft as reference

# numpy implements the FFT up to a phase factor
tmp = np.arange(n)
tmp0, tmp1, tmp2 = np.meshgrid(tmp, tmp, tmp, indexing = 'ij')
phase_corr = ((-1)**tmp0) * ((-1)**tmp1) * ((-1)**tmp2)

res2_numpy = (np.fft.fftn(fk)*phase_corr).ravel()

#----------------------------------------------------------------------

# calculate DFT directly (slow and naive)

tmp = np.arange(n) - n//2
x0, x1, x2 = np.meshgrid(tmp, tmp, tmp, indexing = 'ij')
x = np.array([x0,x1,x2]).reshape(3,-1).astype(dtype)

res2_ref = np.zeros_like(res2)

for i in range(k.shape[1]):
    phase = k[:,i] @ x
    res2_ref[i] = (fk.ravel() * np.exp(-1j*isign*phase)).sum()


#----------------------------------------------------------------------

print(res2[:5])
print(res2_sigpy[:5])
print(res2_ref[:5])

import matplotlib.pyplot as plt

fig, ax = plt.subplots(4, 1, figsize = (8,8))
ax[0].plot(res2.real, 'o', label = 'cufinufft')
ax[0].plot(res2_sigpy.real, 'x', label = 'sigpy')
ax[0].plot(res2_ref.real, '.', label = 'direct DFT')
ax[1].plot(res2.real - res2_ref.real, 'o')
ax[1].plot(res2_sigpy.real - res2_ref.real, 'x')

ax[2].plot(res2.imag, 'o')
ax[2].plot(res2_sigpy.imag, 'x')
ax[2].plot(res2_ref.imag, '.')
ax[3].plot(res2.imag - res2_ref.imag, 'o')
ax[3].plot(res2_sigpy.imag - res2_ref.imag, 'x')

ax[0].set_title('real part')
ax[1].set_title('real part - ref real part')
ax[2].set_title('imag part')
ax[3].set_title('imag part - ref imag part')

ax[0].legend()

fig.tight_layout()
fig.show()

