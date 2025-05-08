#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// This kernel computes the upper triangular matrix multiplication (C = A * B) using warp-level primitives
// to reduce the dot product for each output element. Each block is launched with 32 threads (one warp) that 
// collaboratively compute one element of C. The reduction over the dot product is performed using __shfl_down_sync,
// eliminating the need for shared memory reduction.

__global__ void warp_reduce_upper_triangular_kernel(const float* __restrict__ A,
                                                      const float* __restrict__ B,
                                                      float* __restrict__ C,
                                                      int N) {
    // Map each block to one output element: row = blockIdx.y, col = blockIdx.x
    int row = blockIdx.y;
    int col = blockIdx.x;
    
    // Ensure indices are in bounds and only compute upper triangular elements
    if (row >= N || col >= N || row > col) return;

    float partial = 0.0f;
    int lane = threadIdx.x; // Each block is one warp (32 threads)

    // Each warp computes a dot product: sum_{k=row}^{col} A[row,k]*B[k,col]
    // Each thread in the warp processes a portion of the k-range with a stride of warpSize (32)
    for (int k = row + lane; k <= col; k += 32) {
        partial += A[row * N + k] * B[k * N + col];
    }

    // Perform warp-level reduction using shfl_down_sync
    for (int offset = 16; offset > 0; offset /= 2) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }

    // The first lane writes the final result
    if (lane == 0) {
        C[row * N + col] = partial;
    }
}

// Host function that wraps the kernel launch
// The grid is configured with dimensions (N, N), where each block computes one element of the output
// and blocks corresponding to lower triangular indices (row > col) simply exit.

torch::Tensor warp_reduce_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    // Each block consists of a single warp (32 threads) and grid dimensions cover NxN elements.
    dim3 grid(N, N);
    dim3 block(32);

    warp_reduce_upper_triangular_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_reduce_upper_triangular_matmul, "Warp-level reduction optimized upper triangular matrix multiplication");
}
