#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// Each warp computes one element of the output matrix C.
// Threads within the warp cooperatively load chunks of the K-dimension and use warp-level reduction to accumulate the dot product.

template <typename scalar_t>
__global__ void matmul_cuda_kernel(const scalar_t* __restrict__ A,
                                     const scalar_t* __restrict__ B,
                                     scalar_t* __restrict__ C,
                                     int M, int K, int N) {
    // Determine the number of warps per block along x and y dimensions.
    int warps_per_block_x = blockDim.x / WARP_SIZE;  // e.g., for blockDim.x=128, 128/32 = 4
    int warps_per_block_y = blockDim.y / WARP_SIZE;  // e.g., for blockDim.y=128, 128/32 = 4

    // Calculate the warp's (i, j) position within the output matrix.
    int warp_x = threadIdx.x / WARP_SIZE;  // warp column index within the block
    int warp_y = threadIdx.y / WARP_SIZE;  // warp row index within the block

    int output_row = blockIdx.y * warps_per_block_y + warp_y;
    int output_col = blockIdx.x * warps_per_block_x + warp_x;

    if (output_row < M && output_col < N) {
        int lane = threadIdx.x % WARP_SIZE; // lane index within the warp
        scalar_t sum = 0;
        
        // Each thread in the warp processes a portion of the K-dimension in strides of the warp size.
        for (int k = lane; k < K; k += WARP_SIZE) {
            scalar_t a_elem = A[output_row * K + k];
            scalar_t b_elem = B[k * N + output_col];
            sum += a_elem * b_elem;
        }
        
        // Warp-level reduction using __shfl_down_sync()
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // The first lane in the warp writes the final result
        if (lane == 0) {
            C[output_row * N + output_col] = sum;
        }
    }
}

// Forward function exposed to PyTorch
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());

    // Define block dimensions. Here, we use a block of 128x128 threads, which yields a 4x4 grid of warps per block.
    dim3 threads_per_block(128, 128);
    int warps_per_block_x = threads_per_block.x / WARP_SIZE;  // 4
    int warps_per_block_y = threads_per_block.y / WARP_SIZE;  // 4

    // Compute grid dimensions in terms of warps (each warp computes one output element).
    int grid_x = (N + warps_per_block_x - 1) / warps_per_block_x;
    int grid_y = (M + warps_per_block_y - 1) / warps_per_block_y;
    dim3 num_blocks(grid_x, grid_y);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_cuda_kernel", ([&] {
        matmul_cuda_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}

// Pybind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication forward using warp-level reductions (CUDA)");
}
