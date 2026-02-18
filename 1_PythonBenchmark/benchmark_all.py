import numpy as np
import time
import torch
import multiprocessing as mp
from numba import jit

# 1. Pure Python Implementation
def python_sum(n):
    res = 0
    for i in range(n):
        res += i * i
    return res

# 2. Numba (Compiled - Stand-in for Cython)
@jit(nopython=True)
def compiled_sum(n):
    res = 0
    for i in range(n):
        res += i * i
    return res

def run_benchmarks():
    N = 100_000_000  # Size of task
    results = {}

    print(f"Starting Benchmarks on i5-8600K & 1070 Ti...")

    # --- SINGLE CORE PYTHON ---
    start = time.time()
    python_sum(N)
    results['Python (Single)'] = time.time() - start

    # --- MULTI CORE PYTHON ---
    start = time.time()
    cores = mp.cpu_count()
    with mp.Pool(cores) as p:
        p.map(python_sum, [N // cores] * cores)
    results['Python (All Cores)'] = time.time() - start

    # --- NUMPY (Vectorized) ---
    start = time.time()
    arr = np.arange(N, dtype=np.float64)
    np.sum(arr**2)
    results['NumPy (Single/Multi)'] = time.time() - start

    # --- COMPILED (Numba/Cython equivalent) ---
    # First call is "warmup" (compilation time)
    compiled_sum(1)
    start = time.time()
    compiled_sum(N)
    results['Compiled (C-Speed)'] = time.time() - start

    # --- GPU (PyTorch) ---
    # We use a larger task for GPU to overcome data transfer overhead
    if torch.cuda.is_available():
        # Large matrix mult is better for GPU showcase
        mat_size = 5000
        a = torch.randn(mat_size, mat_size).cuda()
        b = torch.randn(mat_size, mat_size).cuda()
        
        # Warmup
        torch.mm(a, b)
        torch.cuda.synchronize()
        
        start = time.time()
        torch.mm(a, b)
        torch.cuda.synchronize() # Wait for GPU to finish
        results['GPU (1070 Ti)'] = time.time() - start
    
    # Print Results
    print("\n--- RESULTS ---")
    for name, t in results.items():
        print(f"{name:25}: {t:.6f}s")

if __name__ == "__main__":
    run_benchmarks()