#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 32

__global__ void matmul_uniform_flow_kernel(const float* __restrict__ A, 
                                         const float* __restrict__ B, 
                                         float* __restrict__ C, 
                                         const int M, const int N, const int K) {
    // Calculate global thread position
    const int warp_id = (threadIdx.x + threadIdx.y * blockDim.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Calculate output position
    const int row = blockIdx.y * BLOCK_SIZE + warp_id;
    const int col = blockIdx.x * BLOCK_SIZE + lane_id;
    
    // Pre-compute validity masks
    const bool valid_row = row < M;
    const bool valid_col = col < N;
    const bool valid_thread = valid_row && valid_col;
    
    // Initialize accumulator
    float acc = 0.0f;
    
    // Compute K-dimension boundaries for this thread
    const int k_start = 0;
    const int k_end = (K + WARP_SIZE - 1) & ~(WARP_SIZE - 1); // Round up to nearest multiple of WARP_SIZE
    
    // Main computation loop - all threads execute the same number of iterations
    #pragma unroll 4
    for (int k = k_start; k < k_end; k += WARP_SIZE) {
        // Load data - use predication instead of branching
        float a_val = (valid_row && k < K) ? __ldg(&A[row * K + k]) : 0.0f;
        float b_val = (valid_col && k < K) ? __ldg(&B[col * K + k]) : 0.0f;
        
        // Accumulate product
        acc += a_val * b_val;
    }
    
    // Warp-level reduction - all threads participate regardless of validity
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    
    // Only the first thread in each warp writes the result
    // Use predication instead of branching
    if (lane_id == 0 && valid_thread) {
        C[row * N + col] = acc;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Configure kernel launch parameters
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_uniform_flow_kernel<<<grid, block>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B using uniform control flow (CUDA)");
}