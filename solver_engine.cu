#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#define BLOCK 16
#define FULL_MASK 0xffffffff

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

__device__ float g_maxdiff;

// --- Device Helper: Warp Shuffle Reduction ---
__device__ inline float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

// --- 1. Baseline: Global Memory Kernel (Evaluated Every Iteration) ---
__global__ void heatKernelGlobal(const float* __restrict__ T, float* __restrict__ Tnew, int N, bool checkDiff)
{
    extern __shared__ float sdata[];
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float diff = 0.0f;
    if (row <= N - 2 && col <= N - 2) {
        int idx = row * N + col;
        float up    = T[idx - N];
        float down  = T[idx + N];
        float left  = T[idx - 1];
        float right = T[idx + 1];

        float newval = 0.25f * (up + down + left + right);
        Tnew[idx] = newval;
        if (checkDiff) diff = fabsf(newval - T[idx]);
    }

    if (checkDiff) {
        sdata[tid] = diff;
        __syncthreads();
        for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            __syncthreads();
        }
        if (tid == 0) atomicMax((int*)&g_maxdiff, __float_as_int(sdata[0]));
    }
}

// --- 2. Baseline: Shared Memory Tiled Kernel (Halo Padded) ---
__global__ void heatKernelShared(const float* __restrict__ T, float* __restrict__ Tnew, int N, bool checkDiff)
{
    extern __shared__ float smemAll[];
    int tileDim = blockDim.x + 2;
    float* tile = smemAll;
    float* sdata = smemAll + tileDim * tileDim;

    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    int colc = min(col, N - 1);
    int rowc = min(row, N - 1);

    tile[ly * tileDim + lx] = T[rowc * N + colc];
    if (threadIdx.x == 0) tile[ly * tileDim + 0] = T[rowc * N + max(col - 1, 0)];
    if (threadIdx.x == blockDim.x - 1) tile[ly * tileDim + (lx + 1)] = T[rowc * N + min(col + 1, N - 1)];
    if (threadIdx.y == 0) tile[0 * tileDim + lx] = T[max(row - 1, 0) * N + colc];
    if (threadIdx.y == blockDim.y - 1) tile[(ly + 1) * tileDim + lx] = T[min(row + 1, N - 1) * N + colc];
    __syncthreads();

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    float diff = 0.0f;
    if (row <= N - 2 && col <= N - 2) {
        float up    = tile[(ly - 1) * tileDim + lx];
        float down  = tile[(ly + 1) * tileDim + lx];
        float left  = tile[ly * tileDim + (lx - 1)];
        float right = tile[ly * tileDim + (lx + 1)];
        float newval = 0.25f * (up + down + left + right);
        Tnew[row * N + col] = newval;
        if (checkDiff) diff = fabsf(newval - tile[ly * tileDim + lx]);
    }

    if (checkDiff) {
        sdata[tid] = diff;
        __syncthreads();
        for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            __syncthreads();
        }
        if (tid == 0) atomicMax((int*)&g_maxdiff, __float_as_int(sdata[0]));
    }
}

// --- 3. Optimized: Warp-Shuffle Accelerated Kernel with Periodic Checks ---
__global__ void heatKernelOptimized(const float* __restrict__ T, float* __restrict__ Tnew, int N, bool checkDiff)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;
    __shared__ float warpMax[8];

    float diff = 0.0f;
    if (row <= N - 2 && col <= N - 2) {
        int idx = row * N + col;
        float up    = T[idx - N];
        float down  = T[idx + N];
        float left  = T[idx - 1];
        float right = T[idx + 1];
        float newval = 0.25f * (up + down + left + right);
        Tnew[idx] = newval;
        if (checkDiff) diff = fabsf(newval - T[idx]);
    }

    if (checkDiff) {
        diff = warpReduceMax(diff);
        if (lane == 0) warpMax[wid] = diff;
        __syncthreads();

        if (wid == 0) {
            float bdiff = (lane < (blockDim.x * blockDim.y / 32)) ? warpMax[lane] : 0.0f;
            bdiff = warpReduceMax(bdiff);
            if (lane == 0) atomicMax((int*)&g_maxdiff, __float_as_int(bdiff));
        }
    }
}

// --- 4. Algorithmic Leap: Red-Black Gauss-Seidel (RB-GS) SOR Kernel ---
__global__ void heatKernelRBSOR(float* __restrict__ T, int N, float omega, int color, bool checkDiff)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;
    __shared__ float warpMax[8];

    float diff = 0.0f;
    if (row <= N - 2 && col <= N - 2) {
        if (((row + col) & 1) == color) {
            int idx = row * N + col;
            float oldval = T[idx];
            float avg = 0.25f * (T[idx - N] + T[idx + N] + T[idx - 1] + T[idx + 1]);
            float newval = (1.0f - omega) * oldval + omega * avg;
            T[idx] = newval;
            if (checkDiff) diff = fabsf(newval - oldval);
        }
    }

    if (checkDiff) {
        diff = warpReduceMax(diff);
        if (lane == 0) warpMax[wid] = diff;
        __syncthreads();

        if (wid == 0) {
            float bdiff = (lane < (blockDim.x * blockDim.y / 32)) ? warpMax[lane] : 0.0f;
            bdiff = warpReduceMax(bdiff);
            if (lane == 0) atomicMax((int*)&g_maxdiff, __float_as_int(bdiff));
        }
    }
}

struct RunStats {
    long long iterations;
    float time_ms;
    bool converged;
    float max_diff;
    std::vector<float> grid;
};

void initGrid(std::vector<float>& g, int N, float top, float bottom, float left, float right, float initVal) {
    g.assign((size_t)N * N, initVal);
    for (int j = 0; j < N; j++) {
        g[0 * N + j] = top;
        g[(N - 1) * N + j] = bottom;
    }
    for (int i = 0; i < N; i++) {
        g[i * N + 0] = left;
        g[i * N + (N - 1)] = right;
    }
}

RunStats executeSolver(int N, float tol, long long maxIter, int solverType, int checkInterval, float omega) {
    RunStats stats;
    std::vector<float> h_T;
    initGrid(h_T, N, 100.0f, 0.0f, 75.0f, 50.0f, 0.0f);

    size_t bytes = (size_t)N * N * sizeof(float);
    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_T.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_T.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK, BLOCK);
    int interior = N - 2;
    dim3 grid((interior + BLOCK - 1) / BLOCK, (interior + BLOCK - 1) / BLOCK);

    size_t sharedBytesGlobal = block.x * block.y * sizeof(float);
    size_t tileDim = block.x + 2;
    size_t sharedBytesShared = tileDim * tileDim * sizeof(float) + block.x * block.y * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float* dA = d_A;
    float* dB = d_B;
    long long iter = 0;
    float currentDiff = 1e30f;
    float zero = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));

    while (currentDiff > tol && iter < maxIter) {
        bool check = ((iter + 1) % checkInterval == 0) || (iter == 0);
        if (check) CUDA_CHECK(cudaMemcpyToSymbol(g_maxdiff, &zero, sizeof(float)));

        if (solverType == 0) { // Baseline Global
            heatKernelGlobal<<<grid, block, sharedBytesGlobal>>>(dA, dB, N, check);
            std::swap(dA, dB);
        } else if (solverType == 1) { // Baseline Shared
            heatKernelShared<<<grid, block, sharedBytesShared>>>(dA, dB, N, check);
            std::swap(dA, dB);
        } else if (solverType == 2) { // Optimized Shuffle + Batched
            heatKernelOptimized<<<grid, block>>>(dA, dB, N, check);
            std::swap(dA, dB);
        } else if (solverType == 3) { // Red-Black Gauss-Seidel SOR
            heatKernelRBSOR<<<grid, block>>>(dA, N, omega, 0, check);
            heatKernelRBSOR<<<grid, block>>>(dA, N, omega, 1, check);
        }

        if (check) {
            CUDA_CHECK(cudaMemcpyFromSymbol(&currentDiff, g_maxdiff, sizeof(float)));
            if (isnan(currentDiff) || isinf(currentDiff)) {
                currentDiff = 1e30f;
                break; // Divergence detected
            }
        }
        iter++;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    stats.iterations = iter;
    stats.time_ms = ms;
    stats.converged = (currentDiff <= tol);
    stats.max_diff = currentDiff;
    stats.grid.assign((size_t)N * N, 0.0f);
    CUDA_CHECK(cudaMemcpy(stats.grid.data(), dA, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    return stats;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 256;
    int solver = (argc > 2) ? atoi(argv[2]) : 0;
    int interval = (argc > 3) ? atoi(argv[3]) : 1;
    float omega = (argc > 4) ? (float)atof(argv[4]) : 1.0f;
    float tol = 1e-4f;
    long long maxIter = 2000000;

    RunStats res = executeSolver(N, tol, maxIter, solver, interval, omega);
    printf("CSV_OUT,%d,%d,%d,%.5f,%lld,%.4f,%d,%e\n",
           N, solver, interval, omega, res.iterations, res.time_ms, res.converged ? 1 : 0, res.max_diff);
    return 0;
}
