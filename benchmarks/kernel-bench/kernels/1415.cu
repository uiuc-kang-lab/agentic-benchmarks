#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define MAX_DIAG_SIZE 16384  // 64KB / sizeof(float)

// Constant memory for diagonal matrix
__constant__ float d_diag[MAX_DIAG_SIZE];

// CUDA kernel using constant memory and vectorized loads
__global__ void diag_matmul_kernel_constant_mem(
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int lane = threadIdx.x % WARP_SIZE;
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;

    if (warpId < N) {
        const float a_val = d_diag[warpId];  // Load from constant memory
        const int row_offset = warpId * M;

        // Use vectorized memory operations if M is divisible by 4
        if (M % 4 == 0) {
            int vec_M = M / 4;
            for (int vec = lane; vec < vec_M; vec += WARP_SIZE) {
                int idx = row_offset / 4 + vec;
                float4 b_val = reinterpret_cast<const float4*>(B)[idx];
                float4 c_val;
                c_val.x = a_val * b_val.x;
                c_val.y = a_val * b_val.y;
                c_val.z = a_val * b_val.z;
                c_val.w = a_val * b_val.w;
                reinterpret_cast<float4*>(C)[idx] = c_val;
            }
        } else {
            // Fallback to scalar operations
            for (int col = lane; col < M; col += WARP_SIZE) {
                int idx = row_offset + col;
                C[idx] = a_val * B[idx];
            }
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");
    TORCH_CHECK(A.size(0) <= MAX_DIAG_SIZE, "Diagonal matrix too large for constant memory");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    auto C = torch::empty({N, M}, B.options());

    // Copy diagonal matrix to constant memory
    cudaMemcpyToSymbol(d_diag, A.data_ptr<float>(), N * sizeof(float));

    // Configure kernel
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / WARP_SIZE;
    int blocks = (N + warpsPerBlock - 1) / warpsPerBlock;

    diag_matmul_kernel_constant_mem<<<blocks, threadsPerBlock>>>(
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication using constant memory");
}