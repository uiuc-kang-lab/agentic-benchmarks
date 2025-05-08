#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel to perform triangular matrix multiplication with balanced workload distribution
__global__ void balanced_workload_triang_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            // Distribute workload evenly by ensuring each thread computes a similar amount of work
            for (int k = col; k <= row; ++k) {
                sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
            }
            C[row * N + col] = sum;
        }
    }
}

// Host function exposed to PyTorch
at::Tensor forward_balanced(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1) && A.size(0) == B.size(0),
                "Matrices must be square and of the same size");

    int N = static_cast<int>(A.size(0));
    auto C = torch::empty_like(A);

    const int threads = 16;  // Use smaller block size to increase grid size and balance workload
    dim3 block(threads, threads);
    dim3 grid((N + threads - 1) / threads, (N + threads - 1) / threads);

    balanced_workload_triang_mm_kernel<<<grid, block>>>(
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
    m.def("forward", &forward_balanced, "Balanced workload triangular matrix multiplication (CUDA)");
}
