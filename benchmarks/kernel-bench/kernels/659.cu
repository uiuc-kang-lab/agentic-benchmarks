#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// Each warp cooperatively computes one element of C using warp-level reduction
// Instead of using shared memory, threads within a warp partition the dot product along the K dimension

template <typename scalar_t>
__global__ void warp_matmul_kernel(const scalar_t* __restrict__ A,
                                     const scalar_t* __restrict__ B,
                                     scalar_t* __restrict__ C,
                                     int M, int K, int N) {
    // Each block is configured with blockDim.x == WARP_SIZE (32) and blockDim.y = number of warps per block
    // Each warp (indexed by threadIdx.y) computes one element of C
    int lane = threadIdx.x;  // Lane index within the warp [0, 31]
    int warp_id = threadIdx.y; // Which warp in the block

    // Map each warp to one output element: row computed from blockIdx.y and warp id; col from blockIdx.x
    int row = blockIdx.y * blockDim.y + warp_id;
    int col = blockIdx.x;  // gridDim.x is set to cover the N dimension

    scalar_t sum = 0;
    if (row < M && col < N) {
        // Each thread in the warp processes a subset of the K dimension
        for (int k = lane; k < K; k += WARP_SIZE) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        // Perform warp-level reduction using shuffle operations
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // Lane 0 of each warp writes the result
        if (lane == 0) {
            C[row * N + col] = sum;
        }
    }
}

// Forward function: verifies input tensors, sets up grid dimensions, and launches the kernel
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());

    // Configure block dimensions: blockDim.x is warp size and blockDim.y is the number of warps per block
    // Here, we choose 8 warps per block (i.e., 8 output rows computed per block)
    int warps_per_block = 8;
    dim3 threadsPerBlock(WARP_SIZE, warps_per_block);

    // Grid dimensions: grid.x covers the N columns; grid.y covers the M rows in steps of warps_per_block
    dim3 gridDim(N, (M + warps_per_block - 1) / warps_per_block);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "warp_matmul_kernel", ([&] {
        warp_matmul_kernel<scalar_t><<<gridDim, threadsPerBlock>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}

// Binding code with pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication forward (CUDA) with warp-level primitives");
}
