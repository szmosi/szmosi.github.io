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
        {% include figure.liquid loading="eager" path="assets/img/projects/speed_lab/benchmark_results_all.png" title="Project Banner" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## 🏎️ The Scaling Challenge

How do you make Python 100x faster? Most developers know Python is "slow" for loops, but few understand which tool to pick for specific workloads. I stress-tested five different acceleration methods across three distinct mathematical domains, scaling from **$10^6$ to $10^9$ operations.**

---

## 🛠️ Technology Stack

To find the performance frontier, I compared the following frameworks:

* **Vanilla Python & Multiprocessing:** Establishing the raw baseline.
* **NumPy:** Leveraging vectorized operations and C-optimized backends.
* **Numba:** Using Just-In-Time (JIT) compilation to transform Python into machine code.
* **Cython:** Compiling Python-like code into C extensions for raw execution speed.
* **PyTorch (CUDA):** Offloading massive parallel workloads to the GPU.

---

## 📊 Stress Test: Matrix Multiplication (MatMul)

MatMul is the backbone of Deep Learning. This benchmark shows how the gap between CPU and GPU widens exponentially as the matrix grows from a standard scale to a "Big Data" scale.

| Matrix Size | NumPy (CPU) | Numba (JIT) | CUDA (GPU) | Speedup (GPU vs Numba) |
| :--- | :--- | :--- | :--- | :--- |
| **$1,000^2$** | 0.004s | 0.483s | 0.032s | 15x |
| **$4,000^2$** | 0.212s | 1.234s | 0.108s | 11x |
| **$20,000^2$** | 26.01s | 262.50s | **2.42s** | **108.5x** |

> **Key Insight:** At $20,000$ scale, the GPU isn't just faster—it’s the difference between a 2.4-second "blink" and a 4.3-minute wait.

---

## 🔬 Computational Comparison by Task

Not every tool is a "silver bullet." My findings show that the "winner" changes depending on the algorithm type.

### 1. The Iterative King: Mandelbrot ($4,000 \times 4,000$)
For complex, iterative logic where vectorization is difficult:
* **Cython:** 3.55s (Winner for CPU-based logic)
* **GPU:** 3.37s
* **NumPy:** 126.60s (**Huge failure** due to Python loop overhead)

### 2. The Vectorization King: Sum of Squares ($10^9$)
* **Cython:** 0.16s
* **NumPy:** 0.19s
* **Numba:** 0.87s
* **GPU:** 0.59s 

**Wait, why is the GPU slower here?**
This reveals the **"Transfer Tax."** For simple reductions like Sum of Squares, the time it takes to move $10^9$ integers from RAM to VRAM (GPU memory) is longer than the time the CPU takes to just do the math.

---

## 📈 Summary of Efficiency

| Category | Best For... | Why? |
| :--- | :--- | :--- |
| **Py Single/Multi** | Scripting | High overhead; only for logic, not heavy math. |
| **NumPy** | Standard Data Science | Optimized C-routines; king of convenience. |
| **Cython** | Custom Algorithms | Compiles to C; best for complex loops the GPU can't handle. |
| **CUDA (PyTorch)** | Massive Parallelism | Unbeatable for MatMul and high-resolution fractals. |

---

## 🎥 Visualizing the Gap

I developed a real-time animation script that visually demonstrates the "growth" of the Mandelbrot set. This shows how much faster accelerated methods reach the final high-resolution image compared to standard Python loops. 

---

## 💡 Final Thoughts
The project proves that **optimization is context-dependent.** Use NumPy for daily tasks, Cython for custom CPU bottlenecks, and CUDA only when the computational density justifies the data transfer overhead.