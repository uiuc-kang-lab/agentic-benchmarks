#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define maximum number of floats allowed in constant memory for B.
// Adjust this value based on hardware limits. For example, 16384 floats = 64 KB.
#define MAX_B_SIZE 16384

// Declare constant memory for the batched B matrices.
__constant__ float d_B_const[MAX_B_SIZE];

// CUDA kernel for batched matrix multiplication using constant memory for B:
// Computes C = A * B, where A (batch_size, M, K), B (batch_size, K, N), C (batch_size, M, N).
__global__ void bmm_kernel_const(
    const float* __restrict__ A,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * M * N;
    if (idx >= total) return;

    int b = idx / (M * N);
    int rem = idx % (M * N);
    int m = rem / N;
    int n = rem % N;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        float a_val = A[b * M * K + m * K + k];
        float b_val = d_B_const[b * K * N + k * N + n];
        sum += a_val * b_val;
    }
    C[b * M * N + m * N + n] = sum;
}

// Forward function: performs batched matrix multiplication using constant memory for B.
// Expects A and B to be CUDA tensors with shape [batch_size, M, K] and [batch_size, K, N] respectively.
// The B tensor is copied into constant memory for faster, read-only access.

torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    // Ensure the total number of elements in B fits in constant memory.
    int B_elements = batch_size * K * N;
    TORCH_CHECK(B_elements <= MAX_B_SIZE, "B tensor is too large for constant memory");

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    // Copy B's data into constant memory.
    cudaMemcpyToSymbol(d_B_const, B.data_ptr<float>(), B_elements * sizeof(float));

    int total = batch_size * M * N;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;

    bmm_kernel_const<<<blocks, threads>>>(
        A.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication using constant memory for B (CUDA)");
}
