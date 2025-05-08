#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    constexpr int WARP_SIZE = 32;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps_per_row = (N + WARP_SIZE - 1) / WARP_SIZE;
    
    // Calculate which row this warp is processing
    const int row = warp_id / num_warps_per_row;
    const int warp_offset = (warp_id % num_warps_per_row) * WARP_SIZE;
    
    if (row < N) {
        // Calculate the starting column for this warp
        const int col_start = warp_offset + lane_id;
        
        if (col_start < N) {
            // Determine if this warp is entirely above or below the diagonal
            const bool is_above_diagonal = (warp_offset + WARP_SIZE) <= row;
            const bool is_below_diagonal = warp_offset > row;
            
            if (is_above_diagonal) {
                // Process elements fully below diagonal (no divergence)
                float sum = 0.0f;
                #pragma unroll 8
                for (int k = col_start; k <= row; ++k) {
                    sum += A[row * N + k] * B[k * N + col_start];
                }
                C[row * N + col_start] = sum;
            }
            else if (is_below_diagonal) {
                // Process elements fully above diagonal (no divergence)
                C[row * N + col_start] = 0.0f;
            }
            else {
                // Only the diagonal warp needs to handle both cases
                if (col_start <= row) {
                    float sum = 0.0f;
                    #pragma unroll 8
                    for (int k = col_start; k <= row; ++k) {
                        sum += A[row * N + k] * B[k * N + col_start];
                    }
                    C[row * N + col_start] = sum;
                } else {
                    C[row * N + col_start] = 0.0f;
                }
            }
        }
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

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // Calculate grid and block dimensions
    const int threadsPerBlock = 256;
    const int warps_per_row = (N + 31) / 32;
    const int total_warps = N * warps_per_row;
    const int numBlocks = (total_warps * 32 + threadsPerBlock - 1) / threadsPerBlock;

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
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}