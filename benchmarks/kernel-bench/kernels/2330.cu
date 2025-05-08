#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define BLOCK_ROWS 16
#define BLOCK_COLS 16

__global__ void matmul_hybrid_reduction_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    
    // Calculate output position
    int row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    int col = blockIdx.x * BLOCK_COLS + threadIdx.x;
    
    // Initialize shared memory
    sdata[tid] = 0.0f;
    
    float thread_sum = 0.0f;
    
    // Each thread accumulates multiple elements
    for (int k = 0; k < K; k += BLOCK_SIZE) {
        int k_idx = k + tid;
        if (k_idx < K && row < M && col < N) {
            float a_val = __ldg(&A[row * K + k_idx]);
            float b_val = __ldg(&B[col * K + k_idx]);
            thread_sum += a_val * b_val;
        }
    }
    
    // Store in shared memory
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Block-level reduction in shared memory
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();
    
    // Warp-level reduction for final 64 elements
    if (tid < 32) {
        volatile float* smem = sdata;
        if (BLOCK_SIZE >= 64) smem[tid] += smem[tid + 32];
        if (BLOCK_SIZE >= 32) smem[tid] += smem[tid + 16];
        if (BLOCK_SIZE >= 16) smem[tid] += smem[tid + 8];
        if (BLOCK_SIZE >= 8) smem[tid] += smem[tid + 4];
        if (BLOCK_SIZE >= 4) smem[tid] += smem[tid + 2];
        if (BLOCK_SIZE >= 2) smem[tid] += smem[tid + 1];
    }
    
    // Write result
    if (tid == 0 && row < M && col < N) {
        C[row * N + col] = sdata[0];
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    dim3 block(16, 16);  // 256 threads per block
    dim3 grid((N + BLOCK_COLS - 1) / BLOCK_COLS, 
              (M + BLOCK_ROWS - 1) / BLOCK_ROWS);

    matmul_hybrid_reduction_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B using hybrid reduction (CUDA)");
}