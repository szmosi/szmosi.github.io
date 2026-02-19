import os
import numpy as np
import time
import torch
import multiprocessing as mp
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
np.seterr(all='ignore') 




ENABLED_METHODS = {
    "Py Single": True,
    "Py Multi":  True,
    "NumPy":     True,
    "Numba":     True,
    "Cython":    True,
    "GPU":       True
}

try:
    import fast_code
    HAS_CYTHON = True
    print("Cython found!")
except:
    HAS_CYTHON = False
    ENABLED_METHODS["Cython"] = False
    print("Cython not found! Skipping Cython tests.")

# GPU Check
if not torch.cuda.is_available():
    ENABLED_METHODS["GPU"] = False
    print("CUDA GPU not found! Skipping GPU tests.")
else:
    print("CUDA GPU detected and ready.")

# --- MULTIPROCESSING HELPERS ---
# These must be outside the class for Windows compatibility
def py_sum_worker(n):
    res = 0.0
    for i in range(int(n)): res += float(i)**2
    return res

def py_matmul_worker(A_chunk, B):
    size_i = len(A_chunk)
    size_k = len(B)
    size_j = len(B[0])
    res = [[0.0] * size_j for _ in range(size_i)]
    for i in range(size_i):
        for j in range(size_j):
            for k in range(size_k):
                res[i][j] += A_chunk[i][k] * B[k][j]
    return res

def py_mandel_worker(h_start, h_end, w, max_iter):
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

# --- SINGLE CORE / NUMBA FUNCTIONS ---
def py_sum(n):
    res = 0.0
    for i in range(int(n)): res += float(i)**2
    return res

def py_matmul_single(A, B):
    size = len(A)
    res = [[0.0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            for k in range(size):
                res[i][j] += A[i][k] * B[k][j]
    return res

def py_mandel_single(h, w, max_iter):
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

@jit(nopython=True, cache=True)
def numba_sum(n):
    res = 0.0
    for i in range(n): 
        res += float(i)**2
    return res

@jit(nopython=True, cache=True)
def numba_matmul(A, B):
    return np.dot(A, B)

@jit(nopython=True, cache=True)
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

# --- BENCHMARK SUITE ---
class BenchmarkSuite:
    def __init__(self, n=10_000_000, mat_size=400, mandel_dim=(500, 500), mandel_iter=100):
        self.n = n
        self.mat_size = mat_size
        self.m_h, self.m_w = mandel_dim
        self.m_iter = mandel_iter
        self.results = {"Sum": {}, "MatMul": {}, "Mandel": {}}

    def run_sum_benchmarks(self):
        print(f"🚀 Running Sum of Squares ({self.n})...")
        
        if ENABLED_METHODS["Py Single"]:
            print("py single started")
            t0 = time.perf_counter()
            py_sum(self.n)
            self.results['Sum']['Py Single'] = time.perf_counter() - t0

        if ENABLED_METHODS["Py Multi"]:
            print("py multi started")
            t0 = time.perf_counter()
            with mp.Pool(6) as p:
                p.map(py_sum_worker, [self.n//6]*6)
            self.results['Sum']['Py Multi'] = time.perf_counter() - t0

        if ENABLED_METHODS["NumPy"]:
            print("numpy started")
            arr = np.arange(self.n, dtype=np.float64)
            t0 = time.perf_counter()
            np.dot(arr, arr)
            self.results['Sum']['NumPy'] = time.perf_counter() - t0

        if ENABLED_METHODS["Numba"]:
            print("numba started")
            numba_sum(1)
            t0 = time.perf_counter()
            numba_sum(self.n)
            self.results['Sum']['Numba'] = time.perf_counter() - t0

        if ENABLED_METHODS["Cython"]:
            print("cython started")
            if HAS_CYTHON:
                t0 = time.perf_counter()
                fast_code.cython_sum(self.n)
                self.results['Sum']['Cython'] = time.perf_counter() - t0

        if ENABLED_METHODS["GPU"]:
            print("gpu started")
            t0 = time.perf_counter()
            
            # 1. Dynamic chunking: Get free memory and leave a 20% safety buffer
            free_mem = torch.cuda.mem_get_info()[0] * 0.9
            print(free_mem) 
            chunk_size = int(free_mem // 4) # 4 bytes per float32 element
            
            gpu_sum = 0.0
            for start in range(0, self.n, chunk_size):
                end = min(start + chunk_size, self.n)
                
                # 2. Process chunk
                x = torch.arange(start, end, device='cuda', dtype=torch.float32)
                gpu_sum += torch.sum(x * x).item()
                del x
                
            torch.cuda.synchronize()
            self.results['Sum']['GPU'] = time.perf_counter() - t0
            torch.cuda.empty_cache()

    def run_matmul(self):
        print(f"🚀 Benchmarking MatMul ({self.mat_size}x{self.mat_size})...")
        S = self.mat_size
        A_np = np.random.rand(S, S).astype(np.float32)
        B_np = np.random.rand(S, S).astype(np.float32)
        A_list = A_np.tolist()
        B_list = B_np.tolist()

        if ENABLED_METHODS["Py Single"]:
            print("py single started")
            t0 = time.perf_counter()
            py_matmul_single(A_list, B_list)
            self.results["MatMul"]["Py Single"] = time.perf_counter() - t0

        if ENABLED_METHODS["Py Multi"]:
            print("py multi started")
            t0 = time.perf_counter()
            row_chunks = np.array_split(A_list, 6)
            with mp.Pool(6) as p:
                p.starmap(py_matmul_worker, [(chunk.tolist(), B_list) for chunk in row_chunks])
            self.results["MatMul"]["Py Multi"] = time.perf_counter() - t0

        if ENABLED_METHODS["NumPy"]:
            print("numpy started")
            t0 = time.perf_counter()
            np.dot(A_np, B_np)
            self.results["MatMul"]["NumPy"] = time.perf_counter() - t0

        if ENABLED_METHODS["Numba"]:
            print("numba started")
            t0 = time.perf_counter()
            numba_matmul(A_np, B_np) 
            self.results["MatMul"]["Numba"] = time.perf_counter() - t0

        if ENABLED_METHODS["Cython"]:
            print("cython started")
            if HAS_CYTHON:
                res_cy = np.zeros((S, S), dtype=np.float32)
                t0 = time.perf_counter()
                fast_code.cython_matmul(A_np, B_np, res_cy)
                self.results["MatMul"]["Cython"] = time.perf_counter() - t0

        if ENABLED_METHODS["GPU"]:
            print("gpu started")
            mem_per_matrix = self.mat_size**2 * 4
            total_needed = mem_per_matrix * 3 
            
            free_mem = torch.cuda.mem_get_info()[0] * 0.9
            
            if total_needed <= free_mem:
                print("GPU: it fitted")
                A_g = torch.from_numpy(A_np).to('cuda')
                B_g = torch.from_numpy(B_np).to('cuda')
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                
                res = torch.mm(A_g, B_g)
            else:
                print("GPU: it did not fitted, chunking")
                B_g = torch.from_numpy(B_np).to('cuda')
                A_cpu = torch.from_numpy(A_np)
                
                bytes_per_row = self.mat_size * 4 * 2
                row_chunk = int((free_mem - mem_per_matrix) // bytes_per_row)
                row_chunk = max(1, min(row_chunk, self.mat_size)) 
                
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                
                for i in range(0, self.mat_size, row_chunk):
                    A_chunk = A_cpu[i : i + row_chunk].to('cuda')
                    res_chunk = torch.mm(A_chunk, B_g)
            
            torch.cuda.synchronize()
            self.results["MatMul"]["GPU"] = time.perf_counter() - t0
            if 'A_g' in locals(): del A_g
            if 'B_g' in locals(): del B_g
            if 'res' in locals(): del res
            torch.cuda.empty_cache()

    def run_mandel(self):
        print(f"🚀 Benchmarking Mandelbrot ({self.m_h}x{self.m_w})...")
        h, w, iters = self.m_h, self.m_w, self.m_iter

        if ENABLED_METHODS["Py Single"]:
            print("py single started")
            t0 = time.perf_counter()
            py_mandel_single(h, w, iters)
            self.results["Mandel"]["Py Single"] = time.perf_counter() - t0

        if ENABLED_METHODS["Py Multi"]:
            print("py multi started")
            t0 = time.perf_counter()
            chunk_size = h // 6
            ranges = [(i * chunk_size, (i + 1) * chunk_size, w, iters) for i in range(6)]
            with mp.Pool(6) as p:
                p.starmap(py_mandel_worker, ranges)
            self.results["Mandel"]["Py Multi"] = time.perf_counter() - t0
        
        if ENABLED_METHODS["NumPy"]:
            print("numpy started")
            t0 = time.perf_counter()
            y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
            c = x + y*1j
            z = c
            for _ in range(iters):
                z = z**2 + c
            self.results["Mandel"]["NumPy"] = time.perf_counter() - t0

        if ENABLED_METHODS["Numba"]:
            print("numba started")
            numba_mandel(10, 10, 1)
            t0 = time.perf_counter()
            numba_mandel(h, w, iters)
            self.results["Mandel"]["Numba"] = time.perf_counter() - t0

        if ENABLED_METHODS["Cython"]:
            if HAS_CYTHON:
                print("cython started")
                t0 = time.perf_counter()
                fast_code.cython_mandelbrot(h, w, iters)
                self.results["Mandel"]["Cython"] = time.perf_counter() - t0

        if ENABLED_METHODS["GPU"]:
            print("gpu started")
            h, w, iters = self.m_h, self.m_w, self.m_iter

            bytes_per_pixel = 16 
            total_needed = h * w * bytes_per_pixel
            
            free_mem = torch.cuda.mem_get_info()[0] * 0.9
            
            t0 = time.perf_counter()
            
            if total_needed <= free_mem:
                print("GPU: it fitted")
                grid_y = torch.linspace(-1.4, 1.4, h, device='cuda')
                grid_x = torch.linspace(-2, 0.8, w, device='cuda')
                y_g, x_g = torch.meshgrid(grid_y, grid_x, indexing='ij')
                
                c = torch.complex(x_g, y_g)
                z = c.clone()
                for _ in range(iters):
                    z = z * z + c

                del grid_y, grid_x, y_g, x_g, c, z  
            else:
                print("GPU: it did not fitted, chunking")
                grid_x = torch.linspace(-2, 0.8, w, device='cuda')
                grid_y_all = torch.linspace(-1.4, 1.4, h, device='cuda')
                
                row_size_bytes = w * bytes_per_pixel
                row_chunk = int(free_mem // row_size_bytes)
                row_chunk = max(1, min(row_chunk, h))
                
                for i in range(0, h, row_chunk):
                    y_slice = grid_y_all[i : i + row_chunk]
                    y_g, x_g = torch.meshgrid(y_slice, grid_x, indexing='ij')
                    
                    c_chunk = torch.complex(x_g, y_g)
                    z_chunk = c_chunk.clone()
                    for _ in range(iters):
                        z_chunk = z_chunk * z_chunk + c_chunk

                    del y_g, x_g, c_chunk, z_chunk
                
                del grid_x, grid_y_all

            torch.cuda.synchronize()
            self.results["Mandel"]["GPU"] = time.perf_counter() - t0
            torch.cuda.empty_cache()

    def display(self):
        print("\n" + "="*60)
        print(f"{'Category':<15} | {'Method':<20} | {'Time (s)':<10}")
        print("-" * 60)
        for cat, data in self.results.items():
            for method, t in data.items():
                print(f"{cat:<15} | {method:<20} | {t:.6f}s")
        print("="*60)

    def plot_results(self):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        titles = ["Sum of Squares", "Matrix Multiplication", "Mandelbrot Set"]
        for i, (key, data) in enumerate(self.results.items()):
            if not data: continue
            methods = list(data.keys())
            times = list(data.values())
            axes[i].bar(methods, times, color=['#e63946', '#a8dadc', '#457b9d', '#1d3557', '#ffb703', '#8338ec'])
            axes[i].set_title(titles[i])
            axes[i].set_ylabel("Seconds (Log Scale)")
            axes[i].set_yscale('log')
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        output_dir = "Output"
        file_name = "benchmark_results_all.png"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, file_name)
        plt.savefig(save_path)
        print(f"✅ Charts saved to {save_path}")


    def animate_mandelbrot_realtime(self, speed_mult=1.0, res=(500, 500)):
        print("🎬 Generating fractal growth animation...")
        if not os.path.exists('Output'): os.makedirs('Output')
        
        h, w = res
        target_iters = self.m_iter  # Use 1000 from your config
        
        # 1. FIX: Pre-calculate the final high-detail image
        # This is what all 'Done' models will display
        final_image = numba_mandel(h, w, target_iters)

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        axes = axes.flatten()
        methods = ["Py Single", "Py Multi", "NumPy", "Numba", "Cython", "GPU"]
        mandel_times = self.results.get("Mandel", {})
        
        max_bench_time = max(mandel_times.values()) if mandel_times else 10.0
        fps = 20
        total_frames = int((max_bench_time / speed_mult) * fps)
        
        ims, status_texts, timer_texts = [], [], []

        for i, method in enumerate(methods):
            ax = axes[i]
            ax.set_title(f"MODEL: {method}", fontsize=14, fontweight='bold', pad=15)
            ax.set_xticks([]); ax.set_yticks([])
            
            # 2. CRITICAL FIX: Set vmax to target_iters (1000) and NEVER change it
            # This prevents the 'brightening/dimming' flickering effect
            im = ax.imshow(np.zeros((h, w)), cmap='magma', vmin=0, vmax=target_iters)
            ims.append(im)
            
            chk = ax.text(0.95, 0.05, "", transform=ax.transAxes, ha="right", color='lime', fontsize=25)
            status_texts.append(chk)
            
            tmr = ax.text(0.95, 0.92, "0.00s", transform=ax.transAxes, ha="right", 
                        color='white', fontsize=14, fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
            timer_texts.append(tmr)

        def update(frame):
            current_sim_time = (frame / fps) * speed_mult
            
            for i, method in enumerate(methods):
                bench_time = mandel_times.get(method, 0.0)
                
                if current_sim_time >= bench_time:
                    # MODEL IS DONE: Show final sharp image instantly
                    ims[i].set_data(final_image)
                    timer_texts[i].set_text(f"{bench_time:.2f}s")
                    timer_texts[i].set_color('lime')
                    status_texts[i].set_text(r"$\checkmark$")
                    for spine in axes[i].spines.values():
                        spine.set_edgecolor('lime'); spine.set_linewidth(4)
                else:
                    # MODEL IS WORKING: Calculate detail based on progress
                    progress = current_sim_time / bench_time
                    # Gradually increase iterations from 1 up to target_iters
                    current_iters = max(1, int(progress * target_iters))
                    
                    # 3. FIX: Generate the partial frame
                    # Since vmax is locked at 1000, as current_iters increases, 
                    # the image will naturally get brighter and more detailed.
                    partial_image = numba_mandel(h, w, current_iters)
                    ims[i].set_data(partial_image)
                    timer_texts[i].set_text(f"{current_sim_time:.2f}s")
                    
            return ims + status_texts + timer_texts

        ani = FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=False)
        
        save_path = os.path.join("Output", "mandelbrot_growth_replay.mp4")
        try:
            ani.save(save_path, writer='ffmpeg', fps=fps, bitrate=4000)
            print(f"✅ Success! Growth animation saved to {save_path}")
        except:
            ani.save(save_path.replace(".mp4", ".gif"), writer='pillow', fps=fps)
        
        plt.close()



if __name__ == "__main__":
    mp.freeze_support()
    config = {
        "n": 1_000_000_000,
        "mat_size": 1_000,
        "mandel_dim": (1_000, 1_000),
        "mandel_iter": 1_000
    }
    suite = BenchmarkSuite(
        n=config["n"], 
        mat_size=config["mat_size"], 
        mandel_dim=config["mandel_dim"], 
        mandel_iter=config["mandel_iter"]
    )
    
    suite.run_sum_benchmarks()
    suite.run_matmul()
    suite.run_mandel()
    suite.display()
    suite.plot_results()
    #suite.animate_mandelbrot_realtime(speed_mult=2.0, res=(200, 200))