#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 256
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

// CUDA kernel that uses shared memory to load tiles of the B vector.
// Each block computes one row of the matrix-vector multiplication.
template <typename scalar_t>
__global__ void matvec_shared_B_kernel(const scalar_t* __restrict__ A,
                                         const scalar_t* __restrict__ B,
                                         scalar_t* __restrict__ C,
                                         int64_t M, int64_t K) {
    // Each block processes one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    scalar_t sum = 0;

    // Allocate shared memory for one tile of B and for warp-reduction scratch
    __shared__ scalar_t shared_B[TILE_SIZE];
    __shared__ scalar_t sdata_reduce[WARPS_PER_BLOCK];

    // Process the B vector in tiles to reduce global memory accesses
    for (int tile_start = 0; tile_start < K; tile_start += TILE_SIZE) {
        int tile_size = TILE_SIZE;
        if (tile_start + TILE_SIZE > K) {
            tile_size = K - tile_start;
        }
        // Cooperatively load the current tile of B into shared memory
        for (int i = tid; i < tile_size; i += WARP_SIZE) {
            shared_B[i] = B[tile_start + i];
        }
        __syncthreads();

        // Each thread processes a portion of the tile, computing partial dot products
        for (int i = tid; i < tile_size; i += WARP_SIZE) {
            sum += A[row * K + tile_start + i] * shared_B[i];
        }
        __syncthreads();
    }

    // Intra-warp reduction using shuffle instructions
    int lane = tid % WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Write the reduced sum from each warp to shared memory
    if (lane == 0) {
        sdata_reduce[warpId] = sum;
    }
    __syncthreads();

    // First warp reduces the partial sums from each warp
    scalar_t final_sum = 0;
    if (tid < WARPS_PER_BLOCK) {
        final_sum = sdata_reduce[tid];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            final_sum += __shfl_down_sync(0xffffffff, final_sum, offset);
        }
    }

    // Thread 0 writes the final dot product result for this row
    if (tid == 0) {
        C[row] = final_sum;
    }
}

// C++ wrapper for the CUDA kernel
torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1), "B must be a vector");

    auto B_flat = B.view({-1});
    auto C = torch::zeros({M}, A.options());

    dim3 threads(BLOCK_SIZE);
    dim3 blocks(M);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_shared_B_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B_flat.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M,
            K
        );
    }));

    return C.view({M, 1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication using Shared Memory for B (CUDA)");
}
