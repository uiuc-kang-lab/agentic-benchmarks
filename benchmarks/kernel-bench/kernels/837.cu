#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

// Number of warps per block; each warp computes one output element.
// Adjust WARPS_PER_BLOCK to tune occupancy.
#define WARPS_PER_BLOCK 8

// This kernel assigns one warp per output element. Each warp computes the dot product for C[row, col]
// by partitioning the K dimension among its 32 lanes. The __shfl_down_sync intrinsics perform an efficient
// warp-level reduction, eliminating the need for shared memory reductions. This approach is tuned for NVIDIA H100 GPUs.
__global__ void warp_reduce_matmul_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int M, int K, int N) {
    // Each block is configured with 32 x WARPS_PER_BLOCK threads (one warp per output element).
    int lane = threadIdx.x;         // Lane index within a warp [0, 31]
    int warp_id = threadIdx.y;        // Warp id within the block

    // Compute the global row and column indices for the output matrix.
    // Map one warp to one output element, where grid.x covers the column dimension
    // and grid.y*WARPS_PER_BLOCK covers the row dimension.
    int row = blockIdx.y * WARPS_PER_BLOCK + warp_id;
    int col = blockIdx.x;

    float sum = 0.0f;

    if (row < M && col < N) {
        // Each lane in the warp processes a subset of the K dimension with a stride equal to 32 (warp size).
        for (int k = lane; k < K; k += 32) {
            sum += A[row * K + k] * B[k * N + col];
        }

        // Warp-level reduction using __shfl_down_sync to sum the partial results from each lane.
        unsigned mask = 0xffffffff;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }

        // The first lane in the warp writes the final result.
        if (lane == 0) {
            C[row * N + col] = sum;
        }
    }
}

// Host function to launch the kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Allocate output tensor
    auto C = torch::zeros({M, N}, A.options());

    // Launch configuration:
    // - grid.x is set to N (one block per column of C)
    // - grid.y is set to ceil(M / WARPS_PER_BLOCK) (each block computes WARPS_PER_BLOCK rows)
    // - block dimensions are fixed: 32 threads per warp and WARPS_PER_BLOCK warps per block
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid(N, (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    warp_reduce_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Warp-level reduction matrix multiplication (CUDA)");
}
