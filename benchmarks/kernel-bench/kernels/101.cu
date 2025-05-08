#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void matrix_mult_kernel(float *A, float *B, float *C, int M, int N, int K) {
    extern __shared__ float shared_data[]; // Shared memory for tiles
    float *s_A = shared_data;
    float *s_B = shared_data + blockDim.x * blockDim.y;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float C_value = 0;

    for (int t = 0; t < (K + blockDim.x - 1) / blockDim.x; ++t) {
        if (t * blockDim.x + threadIdx.x < K && row < M)
            s_A[threadIdx.y * blockDim.x + threadIdx.x] = A[row * K + t * blockDim.x + threadIdx.x];
        else
            s_A[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;

        if (t * blockDim.x + threadIdx.y < K && col < N)
            s_B[threadIdx.y * blockDim.x + threadIdx.x] = B[(t * blockDim.x + threadIdx.y) * N + col];
        else
            s_B[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;

        __syncthreads(); // Synchronize to ensure tiles are loaded

        for (int i = 0; i < blockDim.x; ++i) {
            C_value += s_A[threadIdx.y * blockDim.x + i] * s_B[i * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = C_value;
    }
}

void matrix_multiply_cuda_atomics(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    // Ensure inputs are CUDA tensors and contiguous
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    // Get the dimensions of the matrices
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Get the pointers to the data
    float *d_A = A.data_ptr<float>();
    float *d_B = B.data_ptr<float>();
    float *d_C = C.data_ptr<float>();

    // Define block size and grid size
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Allocate shared memory
    size_t shared_memory_size = 2 * blockDim.x * blockDim.y * sizeof(float);

    // Launch the kernel
    matrix_mult_kernel<<<gridDim, blockDim, shared_memory_size>>>(d_A, d_B, d_C, M, N, K);

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch kernel: " << cudaGetErrorString(err) << std::endl;
    }
}

torch::Tensor forward_optimized(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are CUDA tensors and contiguous
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    // Get the dimensions of the matrices
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Create the output tensor
    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Perform the matrix multiplication
    matrix_multiply_cuda_atomics(A, B, C);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_optimized", &forward_optimized, "Optimized Matrix multiplication (CUDA)");
}