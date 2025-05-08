#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to process vectorized 4-element chunks
__device__ inline void process_vectorized(const float* B_row, float* C_row, int M, float a, int threadId, int stride) {
    int vec_count = M / 4; // Number of 4-element chunks
    const float4* B_vec = reinterpret_cast<const float4*>(B_row);
    float4* C_vec = reinterpret_cast<float4*>(C_row);
    for (int i = threadId; i < vec_count; i += stride) {
        float4 b_val = B_vec[i];
        float4 c_val;
        c_val.x = a * b_val.x;
        c_val.y = a * b_val.y;
        c_val.z = a * b_val.z;
        c_val.w = a * b_val.w;
        C_vec[i] = c_val;
    }
}

// Device function to process tail elements that cannot be vectorized
__device__ inline void process_tail(const float* B_row, float* C_row, int M, float a, int threadId, int stride) {
    int start = (M / 4) * 4; // Starting index for remaining elements
    for (int i = threadId + start; i < M; i += stride) {
        C_row[i] = a * B_row[i];
    }
}

// Main kernel using modular device functions for improved readability and maintainability
__global__ void diag_matmul_modular_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int row = blockIdx.x; // Each block processes one row
    __shared__ float a_val;

    // Load the diagonal element from A into shared memory
    if (threadIdx.x == 0) {
        a_val = A[row];
    }
    __syncthreads();

    // Pointers to the current row of matrices B and C
    const float* B_row = B + row * M;
    float* C_row = C + row * M;

    int threadId = threadIdx.x;
    int stride = blockDim.x;

    // Process the bulk of the row using vectorized loads/stores
    process_vectorized(B_row, C_row, M, a_val, threadId, stride);

    // Process any remaining tail elements
    process_tail(B_row, C_row, M, a_val, threadId, stride);
}

// Forward function wrapping the CUDA kernel
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");

    // Ensure inputs are contiguous
    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    // Create output tensor
    auto C = torch::empty({N, M}, B.options());

    // Launch one block per row with a fixed number of threads per block
    const int threads = 256;
    diag_matmul_modular_kernel<<<N, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular optimized diagonal matrix multiplication with device functions");
}
