#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Each block is one warp (32 threads) that cooperatively computes one element of C.
// For lower triangular matrices A and B, we compute:
//   C[row, col] = sum_{k=col}^{row} A[row, k] * B[k, col]   for row >= col
// and C[row, col] = 0 for row < col.
// The dot product is partitioned among the 32 threads in a warp, and then a warp-level reduction
// using __shfl_down_sync aggregates the partial sums.

__global__ void warp_triangular_mm_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int N) {
    // Each block is one warp; blockIdx.x equals the global warp id.
    int global_warp_id = blockIdx.x;
    int lane = threadIdx.x; // lane index within the warp (0-31)

    // Total number of elements in an N x N matrix
    int total_elements = N * N;
    if (global_warp_id >= total_elements) return;

    // Map the linear warp id to a 2D index: row = warp_id / N, col = warp_id % N
    int row = global_warp_id / N;
    int col = global_warp_id % N;

    // For upper triangular part, assign zero
    if (row < col) {
        if (lane == 0) {
            C[row * N + col] = 0.0f;
        }
        return;
    }

    // Compute C[row, col] = sum_{k=col}^{row} A[row, k] * B[k, col]
    __shared__ float shared_sum[32];
float lane_sum = 0.0f;
    // Distribute the k-loop among warp lanes: each lane processes indices starting at (col + lane) with a stride of 32
    for (int k = col + lane; k <= row; k += 32) {
        partial_sum += A[row * N + k] * B[k * N + col];
    }

    // Warp-level reduction using __shfl_down_sync to sum the partial sums
    unsigned int mask = 0xFFFFFFFF; // full warp
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }

    // The first lane writes the final result
    if (lane == 0) {
        C[row * N + col] = partial_sum;
    }
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Total number of elements in C
    int total_elements = N * N;  
    // Each block is one warp (32 threads). Launch enough blocks to cover all N*N elements.
    int threads_per_block = 32;
    int num_blocks = (total_elements + 1 - 1) / 1; // one warp (block) per element

    dim3 blocks(num_blocks);
    dim3 threads(threads_per_block);

    warp_triangular_mm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-level triangular matrix multiplication (CUDA)");
}
