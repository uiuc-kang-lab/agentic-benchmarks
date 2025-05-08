#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

// Device function that computes the dot product for a given (row, col) in the upper triangular matrix.
// The loop is manually unrolled to process 4 elements per iteration, reducing loop overhead.
__device__ float compute_unrolled(const float* A, const float* B, int row, int col, int N) {
    float sum = 0.0f;
    int len = col - row + 1;  // Number of multiplications
    int full = len / 4;       // Full groups of 4
    
    // Unroll loop for groups of 4 iterations
    #pragma unroll
    for (int i = 0; i < full * 4; i += 4) {
         int k = row + i;
         sum += A[row * N + k] * B[k * N + col];
         sum += A[row * N + k + 1] * B[(k + 1) * N + col];
         sum += A[row * N + k + 2] * B[(k + 2) * N + col];
         sum += A[row * N + k + 3] * B[(k + 3) * N + col];
    }
    // Handle remaining elements if len is not a multiple of 4
    for (int i = full * 4; i < len; i++) {
         int k = row + i;
         sum += A[row * N + k] * B[k * N + col];
    }
    return sum;
}

// 2D kernel that computes the upper triangular matrix multiplication.
// Each thread processes one element (if valid) in the upper triangular portion of the output matrix.
__global__ void unrolled_upper_triangular_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N && row <= col) {
         C[row * N + col] = compute_unrolled(A, B, row, col, N);
    }
}

// Host function that launches the kernel.
// It creates a threads configuration matching the upper triangular region using a 2D grid.
torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    unrolled_upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Upper triangular matrix multiplication with loop unrolling");
}
