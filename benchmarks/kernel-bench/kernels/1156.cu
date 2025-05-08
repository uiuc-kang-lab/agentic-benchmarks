#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel: Each warp computes one output element's dot product using warp-level primitives
template <typename scalar_t>
__global__ void warp_level_dot_matmul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int M, int K, int L) {

    // Determine warp and lane indices
    int warps_per_block = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;       // Which warp in the block
    int lane = threadIdx.x % 32;          // Lane index within the warp

    // Each warp is responsible for one output element in the (M,L) grid
    // Calculate the global output index for this warp
    int out_idx = blockIdx.x * warps_per_block + warp_id;
    int n = blockIdx.y;  // Batch index (first dimension of A, output has shape [N, M, L])

    if (out_idx < M * L) {
        int m = out_idx / L;  // Row index in A's 2nd dimension
        int l = out_idx % L;  // Column index in B's 2nd dimension

        scalar_t sum = 0;
        // Loop over K dimension, each thread in the warp handles a strided portion
        for (int k = lane; k < K; k += 32) {
            // A is of shape [N, M, K] and B is of shape [K, L]
            sum += A[n * (M * K) + m * K + k] * B[k * L + l];
        }
        
        // Warp-level reduction using shuffle down
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // Lane 0 writes the computed dot product
        if (lane == 0) {
            output[n * (M * L) + out_idx] = sum;
        }
    }
}

// Forward function to launch the kernel
void warp_level_dot_matmul_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    // Choose a block size with multiple warps, e.g., 256 threads (8 warps per block)
    const int threads = 256;
    const int warps_per_block = threads / 32;
    int total_outputs = M * L;  // Number of output elements per batch
    int blocks_x = (total_outputs + warps_per_block - 1) / warps_per_block;

    // Grid: x-dimension covers output elements, y-dimension covers the batch (N)
    dim3 grid(blocks_x, N);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "warp_level_dot_matmul_forward", ([&] {
        warp_level_dot_matmul_kernel<scalar_t><<<grid, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            M, K, L);
    }));

    cudaDeviceSynchronize();
}

// C++ interface
torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {

    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be a 3D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");

    int N = A.size(0);
    int M = A.size(1);
    int L = B.size(1);

    auto output = torch::zeros({N, M, L}, A.options());
    warp_level_dot_matmul_forward(A, B, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Warp-level tensor matrix multiplication (CUDA)");
}
