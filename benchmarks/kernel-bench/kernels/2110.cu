#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void warp_optimized_triangular_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N
) {
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane_id = threadIdx.x & 0x1f;  // Lane ID within warp
    const unsigned int warp_id = threadIdx.x >> 5;    // Warp ID within block

    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.0f;
        } else {
            float sum = 0.0f;
            
            // Process elements in chunks of warp size
            #pragma unroll
            for (int k = col; k <= row; k += 32) {
                const int k_idx = k + lane_id;
                if (k_idx <= row) {
                    sum += __ldg(&A[row * N + k_idx]) * __ldg(&B[k_idx * N + col]);
                }
            }

            // Perform warp-level reduction using shuffle operations
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }

            // First thread in each warp writes the result
            if (lane_id == 0) {
                C[row * N + col] = sum;
            }
        }
    }
}

at::Tensor forward(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be of the same size");

    const int N = static_cast<int>(A.size(0));
    auto C = torch::empty_like(A);

    constexpr int threads = 32;
    const dim3 block(threads, threads);
    const dim3 grid((N + threads - 1) / threads, (N + threads - 1) / threads);

    warp_optimized_triangular_mm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel execution failed");
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized triangular matrix multiplication (CUDA)");
}