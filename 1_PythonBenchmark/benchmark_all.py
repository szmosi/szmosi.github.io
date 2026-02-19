import numpy as np
import time
import torch
import multiprocessing as mp
from numba import jit
import matplotlib.pyplot as plt

try:
    import fast_code
    HAS_CYTHON = True
except:
    HAS_CYTHON = False
    print("Cython not found!")

# Helper for MatMul Multiprocessing
def py_matmul_chunk(A_chunk, B):
    size_i = len(A_chunk)
    size_k = len(B)
    size_j = len(B[0])
    res = [[0.0] * size_j for _ in range(size_i)]
    for i in range(size_i):
        for j in range(size_j):
            for k in range(size_k):
                res[i][j] += A_chunk[i][k] * B[k][j]
    return res

# Helper for Mandelbrot Multiprocessing
def py_mandel_chunk(h_start, h_end, w, max_iter):
    h_chunk = h_end - h_start
    output = np.zeros((h_chunk, w), dtype=np.int32)
    for y in range(h_chunk):
        actual_y = h_start + y
        for x in range(w):
            c = complex(-2.0 + (x * 2.8 / w), -1.4 + (actual_y * 2.8 / (h_start + h_chunk)))
            z = 0j
            for i in range(max_iter):
                z = z*z + c
                if abs(z) > 2:
                    output[y, x] = i
                    break
            else:
                output[y, x] = max_iter
    return output

# --- PURE PYTHON IMPLEMENTATIONS ---
def py_sum(n):
    res = 0.0
    for i in range(int(n)): res += float(i)**2
    return res

def py_matmul(A, B):
    size = len(A)
    res = [[0.0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            for k in range(size):
                res[i][j] += A[i][k] * B[k][j]
    return res

def py_mandel(h, w, max_iter):
    output = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            c = complex(-2.0 + (x*2.8/w), -1.4 + (y*2.8/h))
            z = 0j
            for i in range(max_iter):
                z = z*z + c
                if abs(z) > 2:
                    output[y,x] = i
                    break
            else: output[y,x] = max_iter
    return output

# --- NUMBA IMPLEMENTATIONS ---
@jit(nopython=True)
def numba_sum(n):
    res = 0.0
    for i in range(int(n)): res += float(i)**2
    return res

@jit(nopython=True)
def numba_matmul(A, B):
    return np.dot(A, B)

@jit(nopython=True)
def numba_mandel(h, w, max_iter):
    output = np.zeros((h, w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            c_re = -2.0 + (x * 2.8 / w)
            c_im = -1.4 + (y * 2.8 / h)
            z_re, z_im = 0.0, 0.0
            for i in range(max_iter):
                z_re_sq = z_re * z_re
                z_im_sq = z_im * z_im
                if z_re_sq + z_im_sq > 4.0:
                    output[y, x] = i
                    break
                z_im = 2.0 * z_re * z_im + c_im
                z_re = z_re_sq - z_im_sq + c_re
            else:
                output[y, x] = max_iter
    return output

# --- TASK 2: MATRIX MULTIPLICATION (Size = 2500) ---
# We use 2500 because Python Single Core is VERY slow at Matrix Mult
def py_matmul(size):
    # Standard nested loop matrix multiplication (O(n^3))
    # This is intentionally slow to show why we use libraries
    res = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            for k in range(size):
                res[i][j] += 1.0 * 1.0 
    return res

# --- THE BENCHMARK ENGINE ---
class BenchmarkSuite:
    def __init__(self, n=10_000_000, mat_size=500, mandel_dim=(500, 500), mandel_iter=100):
        self.n = n
        self.mat_size = mat_size
        self.m_h, self.m_w = mandel_dim
        self.m_iter = mandel_iter
        self.results = {"Sum": {}, "MatMul": {}, "Mandel": {}}

    def run_sum_benchmarks(self):
        print("Running Sum of Squares...")
        # 1. Python Single
        t0 = time.perf_counter()
        py_sum(self.n)
        self.results['Sum: Python Single'] = time.perf_counter() - t0

        # 2. Python Multi
        t0 = time.perf_counter()
        with mp.Pool(6) as p:
            p.map(py_sum, [self.n//6]*6)
        self.results['Sum: Python Multi'] = time.perf_counter() - t0

        # 3. NumPy
        arr = np.arange(self.n, dtype=np.float64)
        t0 = time.perf_counter()
        np.sum(arr**2)
        self.results['Sum: NumPy'] = time.perf_counter() - t0

        # 4. Numba
        numba_sum(1) # Warmup
        t0 = time.perf_counter()
        numba_sum(self.n)
        self.results['Sum: Numba'] = time.perf_counter() - t0

        # 5. Cython
        if HAS_CYTHON:
            t0 = time.perf_counter()
            fast_code.cython_parallel_sum(self.n)
            self.results['Sum: Cython Multi'] = time.perf_counter() - t0

        # 6. GPU
        t0 = time.perf_counter()
        x = torch.arange(self.n, device='cuda', dtype=torch.float32)
        torch.sum(x * x)
        torch.cuda.synchronize()
        self.results['Sum: GPU (1070Ti)'] = time.perf_counter() - t0

def run_matmul(self):
        print(f"--- Benchmarking MatMul ({self.mat_size}x{self.mat_size}) ---")
        S = self.mat_size
        A_np = np.random.rand(S, S).astype(np.float32)
        B_np = np.random.rand(S, S).astype(np.float32)
        A_list = A_np.tolist()
        B_list = B_np.tolist()

        # 1. Python Single
        t0 = time.perf_counter()
        py_matmul(A_list, B_list)
        self.results["MatMul"]["Py Single"] = time.perf_counter() - t0

        # 2. Python Multi (Row-based splitting)
        t0 = time.perf_counter()
        row_chunks = np.array_split(A_list, 6)
        with mp.Pool(6) as p:
            p.starmap(py_matmul, [(chunk, B_list) for chunk in row_chunks])
        self.results["MatMul"]["Py Multi"] = time.perf_counter() - t0

        # 3. NumPy (Highly optimized C/MKL)
        t0 = time.perf_counter()
        np.dot(A_np, B_np)
        self.results["MatMul"]["NumPy"] = time.perf_counter() - t0

        # 4. Numba
        # Note: Numba's np.dot is essentially NumPy speed
        t0 = time.perf_counter()
        numba_matmul(A_np, B_np) 
        self.results["MatMul"]["Numba"] = time.perf_counter() - t0

        # 5. Cython
        if HAS_CYTHON:
            res_cy = np.zeros((S, S), dtype=np.float32)
            t0 = time.perf_counter()
            fast_code.cython_matmul(A_np, B_np, res_cy)
            self.results["MatMul"]["Cython"] = time.perf_counter() - t0

        # 6. GPU
        A_g, B_g = torch.from_numpy(A_np).cuda(), torch.from_numpy(B_np).cuda()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.mm(A_g, B_g)
        torch.cuda.synchronize()
        self.results["MatMul"]["GPU"] = time.perf_counter() - t0

def run_mandel(self):
    print(f"--- Benchmarking Mandelbrot ({self.m_h}x{self.m_w}) ---")
    h, w, iters = self.m_h, self.m_w, self.m_iter

    # 1. Python Single
    t0 = time.perf_counter()
    py_mandel(h, w, iters)
    self.results["Mandel"]["Py Single"] = time.perf_counter() - t0

    # 2. Python Multi
    t0 = time.perf_counter()
    h_chunks = [h // 6] * 6
    with mp.Pool(6) as p:
        p.starmap(py_mandel, [(hc, w, iters) for hc in h_chunks])
    self.results["Mandel"]["Py Multi"] = time.perf_counter() - t0

    # 3. NumPy (Vectorized math)
    t0 = time.perf_counter()
    y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
    c = x + y*1j
    z = c
    for _ in range(iters):
        z = z**2 + c # This is the NumPy vectorized way
    self.results["Mandel"]["NumPy"] = time.perf_counter() - t0

    # 4. Numba
    numba_mandel(10, 10, 1) # Warmup
    t0 = time.perf_counter()
    numba_mandel(h, w, iters)
    self.results["Mandel"]["Numba"] = time.perf_counter() - t0

    # 5. Cython
    if HAS_CYTHON:
        t0 = time.perf_counter()
        fast_code.cython_mandelbrot(h, w, iters)
        self.results["Mandel"]["Cython"] = time.perf_counter() - t0

    # 6. GPU (PyTorch Vectorized)
    t0 = time.perf_counter()
    grid_y = torch.linspace(-1.4, 1.4, h, device='cuda')
    grid_x = torch.linspace(-2, 0.8, w, device='cuda')
    y_g, x_g = torch.meshgrid(grid_y, grid_x, indexing='ij')
    c_g = torch.complex(x_g, y_g)
    z_g = c_g.clone()
    for _ in range(iters):
        z_g = z_g*z_g + c_g
    torch.cuda.synchronize()
    self.results["Mandel"]["GPU"] = time.perf_counter() - t0

def display(self):
    print("\n" + "="*50)
    print(f"{'Method':<30} | {'Time (s)':<10}")
    print("-" * 50)
    for k, v in self.results.items():
        print(f"{k:<30} | {v:.6f}s")

def plot_results(self):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        titles = ["Sum of Squares", "Matrix Multiplication", "Mandelbrot Set"]
        
        for i, (key, data) in enumerate(self.results.items()):
            if not data: continue
            methods = list(data.keys())
            times = list(data.values())
            
            axes[i].bar(methods, times, color=['#e63946', '#f1faee', '#a8dadc', '#457b9d', '#1d3557', '#ffb703'])
            axes[i].set_title(titles[i])
            axes[i].set_ylabel("Seconds (Log Scale)")
            axes[i].set_yscale('log')
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig("benchmark_results_all.png")
        print("✅ All plots saved to benchmark_results_all.png")

if __name__ == "__main__":
    # Required for Windows to prevent infinite loop when spawning processes
    mp.freeze_support()

    # --- CONFIGURATION AREA ---
    # Adjust these to scale the benchmark difficulty
    config = {
        "n": 100_000_000,        # Sum of Squares size
        "mat_size": 400,         # Matrix size (Keep < 600 for Pure Python test)
        "mandel_dim": (500, 500),# Resolution of Mandelbrot
        "mandel_iter": 100       # Complexity of Mandelbrot
    }

    # Initialize the Suite
    suite = BenchmarkSuite(
        n=config["n"], 
        mat_size=config["mat_size"], 
        mandel_dim=config["mandel_dim"], 
        mandel_iter=config["mandel_iter"]
    )

    # Execute all 3 Benchmark Categories
    print("Starting Global Benchmark Suite...")
    print(f"Hardware Detected: {mp.cpu_count()} CPU Threads | GPU: {torch.cuda.get_device_name(0)}")
    
    suite.run_sum_benchmarks()   # Runs all 6
    suite.run_matmul()           # Runs all 6
    suite.run_mandel()           # Runs all 6

    # Generate the Comparison Charts
    suite.plot_results()
    suite.display() # Print final table to console