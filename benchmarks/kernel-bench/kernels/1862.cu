#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warp_size;
    const int row = warp_id;

    if (row < N) {
        // Each warp processes one row
        for (int col = lane_id; col < N; col += warp_size) {
            if (col <= row) {
                float sum = 0.0f;
                
                #pragma unroll 4
                for (int k = col; k <= row; k++) {
                    sum += A[row * N + k] * B[k * N + col];
                }
                
                C[row * N + col] = sum;
            } else {
                // Zero out upper triangular part
                C[row * N + col] = 0.0f;
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

    // Configure kernel launch parameters
    const int threads_per_block = 128;  // 4 warps per block
    const int warps_needed = N;
    const int blocks = (warps_needed * 32 + threads_per_block - 1) / threads_per_block;

    triangular_mm_kernel<<<blocks, threads_per_block>>>(
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