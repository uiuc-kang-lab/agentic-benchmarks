#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined optimized kernel using both read-only caching (__ldg) and loop unrolling
__global__ void triangular_mm_kernel_combined(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        // Only compute for the lower triangular part
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            // Loop from k = col to k = row with unrolling directive
            // Use __ldg for read-only access to leverage the read-only cache
            #pragma unroll
            for (int k = col; k <= row; ++k) {
                sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
            }
            C[row * N + col] = sum;
        }
    }
}

// Host function exposed to PyTorch
at::Tensor forward_combined(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be of the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Use a block size that balances warp occupancy and memory throughput
    const int threads = 32;  // using 32 threads per dimension maximizes warp utilization
    dim3 block(threads, threads);
    dim3 grid((N + threads - 1) / threads, (N + threads - 1) / threads);

    triangular_mm_kernel_combined<<<grid, block>>>(
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
    m.def("forward", &forward_combined, "Combined triangular matmul (CUDA) kernel with __ldg and loop unrolling");
}
