#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel using stride loops for better workload distribution
__global__ void triangular_mm_kernel_stride(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = row; i < N; i += stride) {
        for (int j = col; j <= i; j += stride) {
            if (i < N && j < N) {
                float sum = 0.f;
                for (int k = j; k <= i; ++k) {
                    sum += __ldg(&A[i * N + k]) * __ldg(&B[k * N + j]);
                }
                C[i * N + j] = sum;
            }
        }
    }
}

// Host function exposed to PyTorch
at::Tensor forward_stride(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be of the same size");

    int N = static_cast<int>(A.size(0));
    auto C = torch::empty_like(A);

    const int threads = 32;  // using 32 threads per dimension maximizes warp utilization
    dim3 block(threads, threads);
    dim3 grid((N + threads - 1) / threads, (N + threads - 1) / threads);

    triangular_mm_kernel_stride<<<grid, block>>>(
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
    m.def("forward", &forward_stride, "Triangular matrix multiplication with stride loop (CUDA)");
}
