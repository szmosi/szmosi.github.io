import numpy as np
import time
import torch
import multiprocessing as mp
from numba import jit
import matplotlib.pyplot as plt

# Try to import the compiled Cython function
try:
    from fast_code import cython_parallel_sum
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    print("Warning: Cython module not found. Run 'python setup.py build_ext --inplace' first.")

# 1. Pure Python Implementation
def python_sum(n):
    res = 0
    for i in range(int(n)):
        res += i * i
    return res

# 2. Numba (Compiled - Stand-in for Cython Single Core)
@jit(nopython=True)
def numba_sum(n):
    res = 0
    for i in range(int(n)):
        res += i * i
    return res

def run_benchmarks():
    # Use 100 million for CPU to make the gap visible
    N = 1_000_000_000 
    results = {}

    print(f"🚀 Starting Benchmarks on i5-8600K (6C/6T) & GTX 1070 Ti...")

    # --- 1. PYTHON SINGLE CORE ---
    start = time.time()
    python_sum(N)
    results['Python (Single)'] = time.time() - start

    # --- 2. PYTHON ALL CORES ---
    start = time.time()
    cores = mp.cpu_count()
    with mp.Pool(cores) as p:
        p.map(python_sum, [N // cores] * cores)
    results['Python (All Cores)'] = time.time() - start

    # --- 3. NUMPY (Single/Multi) ---
    # Note: NumPy is multi-threaded for math, so it uses your cores automatically
    arr = np.arange(N, dtype=np.float64)
    start = time.time()
    np.sum(arr**2)
    results['NumPy'] = time.time() - start

    # --- 4. NUMBA (C-Speed Single Core) ---
    numba_sum(1) # Warmup
    start = time.time()
    numba_sum(N)
    results['Numba (C-Speed)'] = time.time() - start

    # --- 5. CYTHON MULTICORE ---
    if CYTHON_AVAILABLE:
        start = time.time()
        cython_parallel_sum(N)
        results['Cython (Multi)'] = time.time() - start
    else:
        results['Cython (Multi)'] = 0

    # --- 6. GPU (PyTorch) ---
    if torch.cuda.is_available():
        # Large matrix mult is the best way to show GPU power
        size = 10000 
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')
        torch.cuda.synchronize() # Warmup
        
        start = time.time()
        torch.mm(a, b)
        torch.cuda.synchronize()
        results['GPU (1070 Ti)'] = time.time() - start

    # --- RESULTS TABLE ---
    print("\n" + "="*40)
    print(f"{'Method':<25} | {'Time (s)':<10}")
    print("-"*40)
    for name, t in results.items():
        print(f"{name:<25} | {t:.6f}s")
    print("="*40)

    # --- VISUALIZATION FOR PORTFOLIO ---
    generate_chart(results)

def generate_chart(results):
    # Filter out 0s (if Cython failed)
    plot_data = {k: v for k, v in results.items() if v > 0}
    
    names = list(plot_data.keys())
    times = list(plot_data.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, times, color='skyblue')
    plt.yscale('log') # Log scale because the difference is massive
    plt.ylabel('Time in Seconds (Log Scale)')
    plt.title('Processing Speed Comparison: CPU vs GPU')
    
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}s', va='bottom', ha='center')

    plt.savefig('benchmark_results.png')
    print("\n✅ Success! 'benchmark_results.png' saved for your portfolio.")

if __name__ == "__main__":
    # Multiprocessing needs this on Windows
    mp.freeze_support()
    run_benchmarks()