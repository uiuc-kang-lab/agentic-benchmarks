#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel performs matrix multiplication C = A * B for matrices stored in row-major order.
// It optimizes global memory loads by employing __ldg() for read-only accesses. Additionally, for matrix A,
// loads are vectorized using float4 to perform 128-bit aligned memory accesses when possible.
// In case the inner dimension K is not a multiple of 4, a tail loop handles the remaining elements.

__global__ void matmul_kernel(const float * __restrict__ A, 
                                const float * __restrict__ B, 
                                float * __restrict__ C, 
                                int M, int N, int K) {
    // Calculate row and column indices for the C matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        int k = 0;
        // Process in chunks of 4 to leverage 128-bit loads on A.
        int k_vectorized = (K / 4) * 4; // largest multiple of 4 less than or equal to K
        for (; k < k_vectorized; k += 4) {
            // Load 4 floats from A using vectorized load via __ldg.
            // It is assumed that the starting address A + row*K is 128-bit aligned for most rows.
            float4 a_val = __ldg(reinterpret_cast<const float4*>(A + row * K + k)); // Ensure proper alignment for float4 loads
            // For B, each element is loaded separately since column accesses are not contiguous.
            float b0 = __ldg(B + (k + 0) * N + col);
            float b1 = __ldg(B + (k + 1) * N + col);
            float b2 = __ldg(B + (k + 2) * N + col);
            float b3 = __ldg(B + (k + 3) * N + col);
            sum += a_val.x * b0 + a_val.y * b1 + a_val.z * b2 + a_val.w * b3;
        }
        // Tail loop to process remaining elements in case K is not a multiple of 4
        for (; k < K; k++) {
            float a_val = __ldg(A + row * K + k);
            float b_val = __ldg(B + k * N + col);
            sum += a_val * b_val;
        }
        C[row * N + col] = sum;
    }
}

// Host function for launching the CUDA kernel

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Launch the kernel
    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Optimized Matrix Multiplication (CUDA) with __ldg and 128-bit alignment");
}
