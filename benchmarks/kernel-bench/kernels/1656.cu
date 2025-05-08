#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

// Constant memory for frequently accessed data
__constant__ float const_A[1024 * 1024];  // Adjust size based on expected matrix dimensions

__global__ void upper_triangular_matmul_kernel(const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Only compute upper triangular part
    if (row < N && col < N && row <= col) {
        float sum = 0.0f;

        // Instead of looping over all k, we only need to sum over k from row to col
        int start_k = row;
        int end_k = col;
        int len = end_k - start_k + 1;

        // Pointer to the beginning of the relevant segment in A
        const float* A_ptr = const_A + row * N + start_k;
        // Check if the pointer is 16-byte (128-bit) aligned and we have enough elements for vectorized loads
        bool use_vector = false;
        if (len >= 4 && (((uintptr_t)A_ptr) & 0xF) == 0) {
            use_vector = true;
        }

        int k = start_k;
        // If conditions met, use vectorized loads (float4) for A
        if (use_vector) {
            int vec_iters = len / 4; // number of full 4-element groups
            const float4* A_vec = reinterpret_cast<const float4*>(A_ptr);
            for (int i = 0; i < vec_iters; i++) {
                // Load 4 floats from A using __ldg for read-only cache load
                float4 a_val = A_vec[i];
                int base_k = start_k + i * 4;
                // Multiply each element with corresponding B element (B is accessed in a strided (non-contiguous) manner)
                sum += a_val.x * __ldg(&B[(base_k) * N + col]);
                sum += a_val.y * __ldg(&B[(base_k + 1) * N + col]);
                sum += a_val.z * __ldg(&B[(base_k + 2) * N + col]);
                sum += a_val.w * __ldg(&B[(base_k + 3) * N + col]);
            }
            k = start_k + vec_iters * 4;
        }

        // Process any remaining elements with scalar loads
        for (; k <= end_k; k++) {
            float a_val = const_A[row * N + k];
            float b_val = __ldg(&B[k * N + col]);
            sum += a_val * b_val;
        }
        
        C[row * N + col] = sum;
    }
}

// Host function that wraps the kernel call
torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    // Copy A to constant memory
    cudaMemcpyToSymbol(const_A, A.data_ptr<float>(), N * N * sizeof(float));

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Optimized upper triangular matrix multiplication");
}
