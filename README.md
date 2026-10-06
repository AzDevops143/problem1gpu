# Problem 1: 2D Heat Diffusion with Convergence Detection

## 1. Problem Overview

Consider an $N \times N$ temperature grid representing a two-dimensional physical material. At each discrete iteration $t$, the temperature of each interior cell is updated using the standard 2D finite-difference five-point stencil:

$$T_{i, j}^{(t+1)} = \frac{T_{i-1, j}^{(t)} + T_{i+1, j}^{(t)} + T_{i, j-1}^{(t)} + T_{i, j+1}^{(t)}}{4}$$

where $1 \le i, j \le N - 2$.

---

## 2. Convergence Criterion

The simulation proceeds iteratively until the maximum absolute temperature variation between two consecutive iterations falls strictly below a user-defined tolerance $\varepsilon$:

$$\Delta_{\max}^{(t+1)} = \max_{\substack{1 \le i \le N-2 \\ 1 \le j \le N-2}} \left| T_{i, j}^{(t+1)} - T_{i, j}^{(t)} \right| < \varepsilon$$

- **Parallel Reduction**: The computation of $\Delta_{\max}$ is carried out in parallel directly on the GPU (using block-level shared memory reduction and warp-level shuffle instructions `__shfl_down_sync`).
- **Zero Redundant Transfers**: The entire $N \times N$ temperature grid remains resident in device GPU memory across iterations; only the scalar convergence metric $\Delta_{\max}$ (or a convergence flag) is queried periodically by the host CPU.

---

## 3. Boundary Conditions

The simulation enforces Dirichlet (fixed-temperature) boundary conditions on all four physical edges of the grid:

$$\begin{aligned}
T_{0, j}^{(t)}   &= T_{\text{top}},    \quad && 0 \le j < N \\
T_{N-1, j}^{(t)} &= T_{\text{bottom}}, \quad && 0 \le j < N \\
T_{i, 0}^{(t)}   &= T_{\text{left}},   \quad && 0 \le i < N \\
T_{i, N-1}^{(t)} &= T_{\text{right}},  \quad && 0 \le i < N
\end{aligned}$$

Boundary temperatures remain constant throughout the simulation and are not updated by the stencil kernel.

---

## 4. CUDA Architectural Implementations

This project implements and benchmarks multiple CUDA GPU execution strategies:

1. **Global-Memory Baseline (`heatKernelGlobal`)**:
   - Each thread computes the 5-point stencil by loading its 4 orthogonal neighbors directly from high-latency GPU global memory ($4 \times$ global memory reads per interior cell update).
   - Double-buffering ping-pong pointers (`dA` and `dB`) eliminate data race hazards between iterations.

2. **Shared-Memory Tiled Optimization (`heatKernelShared`)**:
   - Thread blocks load a $(B_x + 2) \times (B_y + 2)$ tile including halo (ghost) boundaries into on-chip shared memory `__shared__`.
   - Stencil computations read exclusively from low-latency, high-bandwidth shared memory, reducing global memory bandwidth pressure.

3. **Warp-Shuffle Batched (`heatKernelOptimized`)**:
   - Warp-level shuffle intrinsics (`warpReduceMax`) accelerate maximum difference reduction.
   - Periodic convergence evaluation (batched over $K$ iterations) eliminates per-iteration kernel launch and PCIe transfer overheads.

4. **Red-Black Gauss-Seidel SOR (`heatKernelRBSOR`)**:
   - Algorithmic acceleration using Red-Black ordering with Successive Over-Relaxation factor $\omega$:
   
   $$\omega_{\text{opt}} = \frac{2}{1 + \sin\left(\frac{\pi}{N}\right)}$$
   
   - Dramatically reduces the number of iterations required to reach $\varepsilon$-convergence.

---

## 5. Input & Output Specification

### Input
| Parameter | Symbol | Description | Typical Values |
| :--- | :---: | :--- | :--- |
| **Grid Dimension** | $N$ | Size of the $N \times N$ grid | $128, 256, 512, 1024$ |
| **Tolerance** | $\varepsilon$ | Stopping criterion threshold | $10^{-4}$ to $10^{-6}$ |
| **Boundary Temps** | $T_{\text{top}}, T_{\text{bottom}}, T_{\text{left}}, T_{\text{right}}$ | Dirichlet boundary temperatures | $100.0, 0.0, 75.0, 50.0\,^\circ\text{C}$ |
| **Relaxation Factor** | $\omega$ | Over-relaxation parameter for SOR | $1.0 \le \omega < 2.0$ |
| **Check Interval** | $K$ | Frequency of convergence checks | $1$ (per-step) to $64$ (batched) |

### Output
- **Converged Temperature Grid**: Final grid state exported to binary/CSV.
- **Iterations Count**: Total iterations required to satisfy $\Delta_{\max} < \varepsilon$.
- **Execution Time**: Kernel computation runtime and throughput measured using CUDA events (`cudaEvent_t`).
- **Throughput**: Measured in Million Cell Updates per second ($\text{MUpdates/sec}$).

---

## 6. Build and Execution

```bash
# Compile with NVCC (optimized -O3)
nvcc -O3 -arch=sm_75 solver_engine.cu -o solver_engine

# Run solver: ./solver_engine <N> <solver_type> <check_interval> <omega>
# Solver Types: 0 = Global, 1 = Shared Tiled, 2 = Warp-Shuffle, 3 = RB-GS SOR

# Example 1: Shared Memory Tiled on N=512
./solver_engine 512 1 1 1.0

# Example 2: Red-Black SOR with optimal omega on N=1024
./solver_engine 1024 3 64 1.99388
```
