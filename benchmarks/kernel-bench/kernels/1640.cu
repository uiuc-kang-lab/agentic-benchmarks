#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

// Maximum number of floats we allow in constant memory (64KB limit / sizeof(float))
#define MAX_CONST_SIZE 16384

// Store matrix B in constant memory for faster read-only access
__constant__ float d_B[MAX_CONST_SIZE];

__global__ void upper_triangular_matmul_kernel(const float* __restrict__ A,
                                                 float* __restrict__ C,
                                                 int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        int start_k = row;
        int end_k = col;
        int len = end_k - start_k + 1;
        const float* a_ptr = A + row * N + start_k;

        // Enable vectorized loads if possible
        bool use_vector = false;
        if (len >= 4 && (((uintptr_t)a_ptr) & 0xF) == 0) {
            use_vector = true;
        }
        int k = start_k;
        if (use_vector) {
            int vec_iters = len / 4;
            const float4* a_vec = reinterpret_cast<const float4*>(a_ptr);
            for (int i = 0; i < vec_iters; i++) {
                float4 a_val = __ldg(&a_vec[i]);
                int base_k = start_k + i * 4;
                // Access B from constant memory d_B
                sum += a_val.x * d_B[base_k * N + col];
                sum += a_val.y * d_B[(base_k + 1) * N + col];
                sum += a_val.z * d_B[(base_k + 2) * N + col];
                sum += a_val.w * d_B[(base_k + 3) * N + col];
            }
            k = start_k + vec_iters * 4;
        }
        // Process any remaining elements
        for (; k <= end_k; k++) {
            sum += __ldg(&A[row * N + k]) * d_B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Host function: copies matrix B into constant memory and launches the kernel
// Assumes that N*N*sizeof(float) <= 64KB (i.e. N <= 128) to fit within hardware limits
torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    TORCH_CHECK(N * N <= MAX_CONST_SIZE, "Matrix B is too large for constant memory usage");

    // Copy matrix B into the constant memory symbol d_B
    cudaMemcpyToSymbol(d_B, B.data_ptr<float>(), N * N * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    auto C = torch::zeros_like(A);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock>>>(A.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Optimized constant memory upper triangular matrix multiplication");
}
