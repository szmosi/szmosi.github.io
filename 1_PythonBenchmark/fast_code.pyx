# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
from cython.parallel import prange
cimport cython

# 1. Sum of Squares
# Handles n up to 10^10 using 64-bit long long
def cython_sum(long long n):
    cdef long long i
    cdef double total = 0.0
    # nogil allows multi-threading via OpenMP
    for i in prange(n, nogil=True):
        total += (<double>i) * (<double>i)
    return total

# 2. Matrix Multiplication (Optimized IKJ Order)
# Swapping j and k loops dramatically improves cache performance
def cython_matmul(float[:,:] A, float[:,:] B, float[:,:] res):
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t size = A.shape[0]
    cdef float temp_a

    # Using 'static' schedule for uniform workloads like matrix math
    for i in prange(size, nogil=True, schedule='static'):
        for k in range(size):
            # Cache A[i, k] so it's not repeatedly looked up in the j-loop
            temp_a = A[i, k]
            for j in range(size):
                # Linear memory access: B[k, j] and res[i, j] are accessed 
                # row-by-row, allowing CPU prefetching and SIMD.
                res[i, j] += temp_a * B[k, j]

# 3. Mandelbrot
# High-resolution fractal generation
def cython_mandelbrot(int h, int w, int max_iter):
    cdef int[::1] output = np.zeros(h * w, dtype=np.int32)
    cdef Py_ssize_t y, x, idx
    cdef int i
    cdef double z_r, z_i, c_r, c_i, z_r_sq, z_i_sq

    for y in prange(h, nogil=True):
        for x in range(w):
            idx = y * w + x
            c_r = -2.0 + (x * 2.8 / w)
            c_i = -1.4 + (y * 2.8 / h)
            z_r = 0.0
            z_i = 0.0
            for i in range(max_iter):
                z_r_sq = z_r * z_r
                z_i_sq = z_i * z_i
                
                if z_r_sq + z_i_sq > 4.0:
                    output[idx] = i
                    break
                
                # Manual expansion of (z^2 + c) for maximum speed
                z_i = 2.0 * z_r * z_i + c_i
                z_r = z_r_sq - z_i_sq + c_r
            else:
                output[idx] = max_iter
                
    return np.array(output).reshape((h, w))