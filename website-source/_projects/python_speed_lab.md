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

## 🏎️ The Scaling Challenge: From $10^6$ to $10^{11}$

Most developers know Python is "slow" for loops, but few understand exactly **where** the bottleneck shifts from the CPU to the RAM, or from the RAM to the GPU. I stress-tested five acceleration methods across three distinct mathematical domains to find the "breaking point" of each technology.

---

## 📊 Stress Test 1: Matrix Multiplication (MatMul)
MatMul is memory-bandwidth intensive. As the matrix grows, the overhead of Python's memory management becomes the primary bottleneck.

### Small vs. Massive Scale Comparison
| Matrix Size | Numba (JIT) | Cython | CUDA (GPU) | Winning Method |
| :--- | :--- | :--- | :--- | :--- |
| **$1,000^2$** | 0.52s | 3.43s | **0.07s** | **GPU** |
| **$4,000^2$** | 1.31s | 87.02s | **0.04s** | **GPU** |
| **$10,000^2$** | 4.46s | 203.02s | **0.31s** | **GPU (Optimized)** |
| **$20,000^2$** | 262.50s | *Timed Out* | **2.42s** | **GPU (108x Speedup)** |

> **💡 The "Cython Collapse" Insight:** Notice Cython's performance at 10k ($203s$). Without manual memory tiling or linking to a BLAS library, Cython struggles with cache-locality in large matrices, whereas Numba and PyTorch handle this automatically.

---

## 🔬 Stress Test 2: Mandelbrot Set (Iterative Logic)
The Mandelbrot set requires heavy complex-number math. Unlike MatMul, this is "compute-bound" rather than just "memory-bound."

### Scaling to $10,000 \times 10,000$ Resolution
| Method | Time (s) | Efficiency Note |
| :--- | :--- | :--- |
| **NumPy (Vectorized)** | 774.56s | Poor; creates massive intermediate arrays in RAM. |
| **Py Multi (8-Core)** | 435.62s | Better, but still bound by Python's interpreter. |
| **Numba (JIT)** | 56.44s | Excellent; fuses loops into machine code. |
| **Cython** | 22.40s | **Best CPU Performance**; raw C-speed. |
| **CUDA GPU** | **19.89s** | **Overall Winner** (when data fits VRAM). |



---

## ⚠️ The "VRAM Ceiling": When the GPU Fails
A critical finding in this lab was the **Memory Overflow** event. When a task is too large for the GPU's onboard video memory (VRAM), the system must "chunk" the data—swapping parts back and forth to the CPU.

### Performance Impact of Memory Chunking
| Scenario ($10k \times 10k$ Mandel) | Hardware State | Execution Time |
| :--- | :--- | :--- |
| **Optimized GPU** | Data fits in VRAM | **19.89s** |
| **Chunked GPU** | **VRAM Overflow** | **327.33s** |
| **Cython (CPU)** | Data in System RAM | **22.03s** |

**Engineering Lesson:** If your dataset is larger than your GPU memory, a well-optimized **Cython** script is actually **14x faster** than a GPU forced to chunk data. 

---

## ⚡ Stress Test 3: Sum of Squares (Extreme Scaling)
To test the "Transfer Tax," I ran a simple reduction (Sum of $x^2$) up to **100 Billion** operations.

| Operations | Numba (JIT) | Cython | CUDA (GPU) |
| :--- | :--- | :--- | :--- |
| **1 Billion** | 0.91s | **0.91s** | 0.76s |
| **2 Billion** | 1.73s | **0.35s** | 2.46s |
| **10 Billion** | 8.62s | **1.53s** | 6.62s |
| **100 Billion** | 86.10s | **14.99s** | 86.13s |

> **The "Transfer Tax" Paradox:** At $10^{11}$ scale, **Cython** outperformed the GPU. Why? Because the GPU spent more time waiting for the CPU to send data over the PCIe bus than it spent actually calculating the sum.

---

## 📈 Final Summary & Technology Map



| Technology | Sweet Spot | Avoid When... |
| :--- | :--- | :--- |
| **NumPy** | Standard Data Science ($<10^6$ elements) | You have nested loops or high-res fractals. |
| **Cython** | Custom C-logic; Limited VRAM | You need rapid prototyping (compilation is slow). |
| **Numba** | NumPy-heavy code needing a boost | Using non-supported Python libraries. |
| **CUDA** | Neural Networks; High-density math | Data transfer is high but math is "simple." |

---

## 🎥 Visualizing the Gap
I developed a real-time animation script that visually demonstrates the "growth" of the Mandelbrot set. It renders the fractal layer-by-layer, showcasing the raw speed of GPU-accelerated frame generation vs. the slow crawl of standard Python.

---

## 💡 Conclusion
The most important takeaway from this lab: **The "fastest" tool depends on the data size.**
* **Small Data:** NumPy/Numba is king.
* **Complex Logic:** Cython is the CPU powerhouse.
* **Massive Parallelism:** CUDA is unbeatable, provided you don't hit the VRAM ceiling.