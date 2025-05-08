#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel with improved thread and block indexing: each thread computes one element of C
// Block dimensions are chosen as 32 x 8 to better leverage GPU resources on the H100
__global__ void ImprovedIndexingKernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int K, int N) {
    // Map each thread to one output element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        // Use __ldg for improved caching on read-only data
        #pragma unroll
        for (int k = 0; k < K; k++) {
            sum += __ldg(&A[row * K + k]) * __ldg(&B[k * N + col]);
        }
        C[row * N + col] = sum;
    }
}

// Host function providing PyTorch binding
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Define block dimensions optimized for H100: 32 threads in x and 8 threads in y
    int blockDimX = 32;
    int blockDimY = 8;
    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim((N + blockDimX - 1) / blockDimX, (M + blockDimY - 1) / blockDimY);

    ImprovedIndexingKernel<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Improved indexing matrix multiplication (CUDA)");
}
