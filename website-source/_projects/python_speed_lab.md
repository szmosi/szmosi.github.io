---
layout: page
title: Python Speed Lab
description: Benchmarking CPU vs. GPU performance across NumPy, Numba, Cython, and CUDA.
img: assets/img/projects/speed_lab/benchmark_results_all.png
importance: 1
category: work
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/speed_lab/benchmark_results_all.png" title="Benchmark Results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## 🏎️ High-Performance Computing Benchmarks

How do you make Python 100x faster? This project explores the performance frontiers of Python, comparing standard execution against modern acceleration frameworks.

---

### 📊 Performance Showdown

I benchmarked three computationally intensive tasks across different scales:

1.  **Sum of Squares:** Large-scale reduction.
2.  **Matrix Multiplication:** Memory-bandwidth and FLOPs intensive.
3.  **Mandelbrot Set:** Iterative complex number math (Fractal generation).

---

### 🚀 Key Results ($20,000 \times 20,000$ Matrix)

| Method | Execution Time | Speedup vs Numba |
| :--- | :--- | :--- |
| **Numba (JIT)** | 262.50s | 1x |
| **NumPy (Vectorized)** | 26.01s | 10.1x |
| **CUDA GPU (PyTorch)** | **2.42s** | **108.5x** |

---

### 🛠️ Technology Stack

* **Vanilla Python & Multiprocessing:** Establishing the baseline.
* **NumPy:** Leveraging vectorized operations and C-optimized backends.
* **Numba:** Using Just-In-Time (JIT) compilation to transform Python into machine code.
* **Cython:** Compiling Python-like code into C extensions for raw execution speed.
* **PyTorch (CUDA):** Offloading massive parallel workloads to the GPU.

---

### 📈 Analysis & Insights

* **The GPU Advantage:** While the GPU wins at scale, for small tasks, NumPy is often faster due to the overhead of moving data from CPU (RAM) to GPU (VRAM).
* **Optimization Sweet Spots:**
    * **Cython** performed exceptionally well on iterative tasks like the Mandelbrot set.
    * **NumPy** remains the king of convenience for standard linear algebra.
    * **GPU acceleration** is the only viable path for "Big Data" scales ($20,000+$ matrices).

### 🎥 Visualizing the Gap

I developed a real-time animation script that visually demonstrates the "Growth" of the Mandelbrot set. This shows how much faster accelerated methods reach the final high-resolution image compared to standard Python loops.