#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define BLOCK_ROWS 8

// Device function to handle boundary conditions
__device__ __forceinline__ bool is_valid_lower_triangular(int row, int col, int N) {
    return (row < N && col < N && row >= col);
}

// Device function to compute tile boundaries
__device__ __forceinline__ void compute_tile_bounds(int t, int row, int col, 
                                                   int* k_start, int* k_end, int N) {
    *k_start = max(t * TILE_SIZE, col);
    *k_end = min((t + 1) * TILE_SIZE, min(row + 1, N));
}

// Device function to load tile from matrix A
__device__ __forceinline__ void load_A_tile(float* shA, const float* A,
                                           int row, int tile_col, int N, int tid) {
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS) {
        int idx = tid + i;
        if (idx < TILE_SIZE) {
            int global_col = tile_col + idx;
            shA[threadIdx.y * TILE_SIZE + idx] = 
                (global_col <= row && global_col < N) ? A[row * N + global_col] : 0.0f;
        }
    }
}

// Device function to load tile from matrix B
__device__ __forceinline__ void load_B_tile(float* shB, const float* B,
                                           int tile_row, int col, int N, int tid) {
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS) {
        int idx = tid + i;
        if (idx < TILE_SIZE) {
            int global_row = tile_row + idx;
            shB[idx * TILE_SIZE + threadIdx.x] = 
                (global_row >= col && global_row < N) ? B[global_row * N + col] : 0.0f;
        }
    }
}

// Device function to compute tile multiplication
__device__ __forceinline__ float compute_tile_product(const float* shA, const float* shB,
                                                     int k_start, int k_end, int t) {
    float sum = 0.0f;
    int local_start = k_start - t * TILE_SIZE;
    int local_end = k_end - t * TILE_SIZE;
    
    if (local_end - local_start == TILE_SIZE) {
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += shA[threadIdx.y * TILE_SIZE + k] * shB[k * TILE_SIZE + threadIdx.x];
        }
    } else {
        #pragma unroll 4
        for (int k = local_start; k < local_end; k++) {
            sum += shA[threadIdx.y * TILE_SIZE + k] * shB[k * TILE_SIZE + threadIdx.x];
        }
    }
    return sum;
}

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float shA[TILE_SIZE * TILE_SIZE];
    __shared__ float shB[TILE_SIZE * TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Early exit for invalid indices or upper triangular part
    if (!is_valid_lower_triangular(row, col, N)) {
        if (row < N && col < N && row < col) {
            C[row * N + col] = 0.0f;
        }
        return;
    }

    float sum = 0.0f;
    int t_start = col / TILE_SIZE;
    int t_end = row / TILE_SIZE;

    #pragma unroll 2
    for (int t = t_start; t <= t_end; t++) {
        // Load tiles collaboratively
        load_A_tile(shA, A, row, t * TILE_SIZE, N, tid);
        load_B_tile(shB, B, t * TILE_SIZE, col, N, tid);
        
        __syncthreads();

        // Compute tile boundaries and perform multiplication
        int k_start, k_end;
        compute_tile_bounds(t, row, col, &k_start, &k_end, N);
        sum += compute_tile_product(shA, shB, k_start, k_end, t);

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(TILE_SIZE, BLOCK_ROWS);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Set L1 cache preference
    cudaFuncSetCacheConfig(triangular_mm_kernel, cudaFuncCachePreferL1);

    triangular_mm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular optimized triangular matrix multiplication (CUDA)");
}