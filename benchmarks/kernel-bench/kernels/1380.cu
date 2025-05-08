#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Threshold for switching between implementations
#define VECTORIZATION_THRESHOLD 512

__global__ void diag_matmul_2d_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < M) {
        C[row * M + col] = A[row] * B[row * M + col];
    }
}

__global__ void diag_matmul_vectorized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    // Process multiple rows per block for better occupancy
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;
    
    __shared__ float a_vals[32];  // Support up to 32 rows per block
    
    if (threadIdx.x == 0) {
        a_vals[threadIdx.y] = A[row];
    }
    __syncthreads();
    
    int col = threadIdx.x;
    const int stride = blockDim.x;
    const float4* B_vec = reinterpret_cast<const float4*>(B + row * M);
    float4* C_vec = reinterpret_cast<float4*>(C + row * M);
    const float a_val = a_vals[threadIdx.y];
    
    const int vec_limit = M / 4;
    while (col < vec_limit) {
        float4 b4 = B_vec[col];
        float4 c4;
        c4.x = a_val * b4.x;
        c4.y = a_val * b4.y;
        c4.z = a_val * b4.z;
        c4.w = a_val * b4.w;
        C_vec[col] = c4;
        col += stride;
    }
    
    col = threadIdx.x + (vec_limit * 4);
    while (col < M) {
        C[row * M + col] = a_val * B[row * M + col];
        col += stride;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0),
                "Dimension mismatch: A.size(0) must match B.size(0)");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    auto C = torch::empty({N, M}, B.options());

    if (M >= VECTORIZATION_THRESHOLD && M % 4 == 0) {
        // Use vectorized version for large matrices with aligned memory
        const int threads = 256;
        diag_matmul_vectorized_kernel<<<N, threads>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            M
        );
    } else {
        // Use 2D grid version for smaller matrices or unaligned memory
        dim3 threads(16, 16);
        dim3 blocks((N + threads.x - 1) / threads.x, 
                   (M + threads.y - 1) / threads.y);
        diag_matmul_2d_kernel<<<blocks, threads>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            M
        );
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive diagonal matrix multiplication");
}