# fast_code.pyx
import numpy as np
cimport numpy as cnp
from cython.parallel import prange

def cython_parallel_sum(int n):
    cdef int i
    cdef long total = 0
    # The 'nogil' and 'prange' allow all 6 threads of your i5 to work at once
    for i in prange(n, nogil=True):
        total += i * i
    return total