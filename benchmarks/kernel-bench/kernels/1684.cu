#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    const unsigned FULL_MASK = 0xffffffff;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= N || col >= N) return;

    // Get active threads in warp
    const unsigned active_mask = __activemask();
    
    // Compute predicate for lower triangular part
    const bool is_lower = (row >= col);
    
    // Synchronize predicate across warp
    const unsigned lower_mask = __ballot_sync(active_mask, is_lower);
    
    float sum = 0.0f;
    
    // All threads participate in computation, results masked later
    #pragma unroll 4
    for (int k = 0; k < N; k++) {
        // Only process elements up to 'row' for each thread
        bool valid_k = (k >= col) && (k <= row);
        unsigned valid_mask = __ballot_sync(active_mask, valid_k);
        
        if (__popc(valid_mask)) {  // If any thread needs this iteration
            float a_val = (valid_k) ? A[row * N + k] : 0.0f;
            float b_val = (valid_k) ? B[k * N + col] : 0.0f;
            sum += a_val * b_val;
        }
    }

    // Write result only if in lower triangle
    if (row < N && col < N) {
        C[row * N + col] = is_lower ? sum : 0.0f;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Use 32x32 thread blocks to match warp size
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + 31) / 32, (N + 31) / 32);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Triangular matrix multiplication with warp synchronization (CUDA)");
}