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
        {% include figure.liquid loading="eager" path="assets/img/projects/speed_lab/benchmark_results_all.png" title="Comprehensive Benchmark Results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## 🏎️ The Quest for 100x Speed: From $10^6$ to $10^{11}$

Python is the language of choice for Data Science, yet its high-level abstraction comes at a massive cost in execution speed. Most developers know Python is "slow" for loops, but few understand exactly **where** the bottleneck shifts from the CPU to the RAM, or from the RAM to the GPU. 

This lab documents a comprehensive stress test of five acceleration methods across three distinct mathematical domains. By pushing these technologies to their "breaking point," I mapped the performance frontiers of modern computing, moving from **Vanilla Python loops** to **CUDA-accelerated kernels**.

---

## 📉 The Baseline: Why We Optimize
To understand the need for acceleration, we must look at "Standard" Python. Even at a modest scale ($1,000 \times 1,000$), the gap between a standard loop and an optimized library is staggering. Python’s Global Interpreter Lock (GIL) and dynamic typing create significant overhead for every single numerical operation.

### Baseline Performance ($1,000 \times 1,000$ Operations)

| Category | Method | Time (s) | Relative Speed |
| :--- | :--- | :--- | :--- |
| **Sum of Squares** | Python Single-Core | 118.61s | 1x (Baseline) |
| **Sum of Squares** | Python Multi-Core | 23.87s | 5x Faster |
| **Sum of Squares** | **NumPy (Vectorized)** | **0.19s** | **624x Faster** |
| **MatMul** | Python Single-Core | 123.13s | 1x (Baseline) |
| **MatMul** | **NumPy (MKL)** | **0.013s** | **9,470x Faster** |


**Conclusion:** For basic math, NumPy is so efficient that "Standard" Python shouldn't even be in the conversation. However, as datasets grow, even NumPy's CPU-bound vectorization reaches its limits.

---

## 📊 Stress Test 1: Matrix Multiplication (MatMul)
MatMul is memory-bandwidth intensive. As the matrix grows, the overhead of Python's memory management becomes the primary bottleneck, and we transition from CPU-based logic to GPU-parallelism.



### Scaling to Big Data Levels

| Matrix Size | NumPy (CPU) | Numba (JIT) | Cython | CUDA (GPU) | Winning Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$1,000^2$** | 0.013s | 0.52s | 3.43s | **0.07s** | **NumPy (MKL)** |
| **$4,000^2$** | 0.21s | 1.31s | 87.02s | **0.04s** | **GPU** |
| **$10,000^2$** | 5.96s | 4.46s | 203.02s | **0.31s** | **GPU (Optimized)** |
| **$20,000^2$** | 26.01s | 262.50s | *Timed Out* | **2.42s** | **GPU (108x Speedup)** |

> **💡 The "Cython Collapse" Insight:** At the $10,000^2$ scale, Cython’s performance collapsed. This illustrates the **"Cache Miss"** problem: without manual memory tiling or specialized linear algebra kernels (like those in NumPy's MKL or PyTorch's CuBLAS), raw C-loops struggle with massive memory blocks.

---

## 🔬 Stress Test 2: Mandelbrot Set (Iterative Logic)
The Mandelbrot set tests how frameworks handle **complex branching logic** that is difficult to vectorize. Unlike MatMul, this is "compute-bound"—the CPU spends more time on arithmetic than fetching data.

### Scaling to $10,000 \times 10,000$ Resolution

| Method | Time (s) | Efficiency Note |
| :--- | :--- | :--- |
| **NumPy (Vectorized)** | 774.56s | Poor; creates massive intermediate arrays in RAM. |
| **Py Multi (8-Core)** | 435.62s | Linear scaling; still bound by Python's interpreter. |
| **Numba (JIT)** | 56.44s | Excellent; fuses loops into machine code. |
| **Cython** | **22.40s** | **CPU King**; Raw C-speed and fine-grained control. |
| **CUDA GPU** | **19.89s** | **Overall Winner** via massive parallelism. |



---

## ⚠️ The "VRAM Ceiling": When the GPU Fails
A critical finding in this lab was the **Memory Overflow** event. If the data is processed in a way that exceeds the GPU's onboard Video RAM (VRAM), the system must "chunk" the data—swapping parts back and forth to the system RAM via the PCIe bus.

### Performance Impact of Memory Chunking

| Scenario (Mandel 10k) | Hardware State | Execution Time |
| :--- | :--- | :--- |
| **Optimized GPU** | Data fits in VRAM | **19.89s** |
| **Chunked GPU** | **VRAM Overflow** | **327.33s** |
| **Cython (CPU)** | Data in System RAM | **22.03s** |


**Engineering Lesson:** In cases of VRAM overflow, the GPU becomes **14x slower** than the CPU. Memory management and data locality are often more important than raw clock speed or core count.

---

## ⚡ Stress Test 3: Sum of Squares (The "Transfer Tax")
To test the limits of the PCIe bus, I ran a simple reduction (Sum of $x^2$) up to **100 Billion** operations. This tests the "Transfer Tax"—the time cost of moving data from CPU RAM to GPU VRAM.



| Operations | Numba (JIT) | Cython | CUDA (GPU) | Winner |
| :--- | :--- | :--- | :--- | :--- |
| **1 Billion** | 0.91s | 0.91s | **0.76s** | **GPU** |
| **10 Billion** | 8.62s | **1.53s** | 6.62s | **Cython** |
| **100 Billion** | 86.10s | **14.99s** | 86.13s | **Cython** |

> **The Transfer Paradox:** At $10^{11}$ scale, **Cython** outperformed the GPU. Why? Because the GPU spent more time waiting for the CPU to send data over the PCIe bus than it spent actually calculating the sum. For simple math on large data, the communication overhead exceeds the computation benefit.

---

## 🎥 Visualizing the Gap: Real-Time Rendering
I developed a series of visualization scripts to demonstrate these benchmarks in real-time. 

* **Fractal Layering:** A real-time animation that renders the Mandelbrot set layer-by-layer. This showcases the raw speed of GPU-accelerated frame generation ($~60$ FPS) vs. the slow crawl of standard Python ($<1$ FPS).
* **Memory Heatmaps:** Visualizes how Numba and Cython access memory vs. the unoptimized "cache-thrashing" of standard Python loops.
* **The Bottleneck Reveal:** A side-by-side video comparison where the CPU progress bar lags significantly behind the GPU until the dataset hits the "VRAM Ceiling," at which point the GPU progress bar visually "stutters" or stops.

---

## 📈 Final Summary & Optimization Map

The most important takeaway: **The "fastest" tool depends entirely on the data size and task complexity.**

| Technology | Sweet Spot | Avoid When... |
| :--- | :--- | :--- |
| **NumPy** | Standard Data Science ($<10^6$ elements) | You have nested loops or high-res fractals. |
| **Cython** | Custom C-logic; Limited VRAM; Extreme Scaling | You need rapid prototyping (compilation is slow). |
| **Numba** | NumPy-heavy code needing a JIT boost | Using non-supported Python libraries/objects. |
| **CUDA** | Neural Networks; High-density math | Data transfer is high but math is "simple." |

---

## 💡 Engineering Conclusion
High-performance Python isn't about using the "fastest" tool; it's about using the **right** tool for the specific hardware constraints. This project proves that while the GPU is a powerhouse for dense parallel math, the CPU (via Cython) remains the king of memory-bound logic and extreme-scale reductions where the PCIe bus becomes a bottleneck.

**Final Rule of Thumb:**
1. **Complex Math + Large Data (Fits VRAM):** Use GPU.
2. **Simple Math + Massive Data (Exceeds VRAM):** Use Cython/CPU.
3. **Rapid Prototyping:** Use NumPy/Numba.