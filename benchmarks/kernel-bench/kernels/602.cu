#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define WARP_SIZE 32

template <typename scalar_t>
__global__ void matmul_coalesced_kernel(const scalar_t* __restrict__ A,
                                       const scalar_t* __restrict__ B,
                                       scalar_t* __restrict__ C,
                                       const int M, const int K, const int N) {
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    const int bx = blockIdx.x * TILE_WIDTH;
    const int by = blockIdx.y * TILE_WIDTH;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Global row and column
    const int row = by + ty;
    const int col = bx + tx;

    scalar_t sum = 0;

    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++tile) {
        // Collaborative loading of A and B tiles into shared memory
        // Each thread in a warp loads consecutive elements
        const int tile_offset = tile * TILE_WIDTH;
        
        if (row < M && tile_offset + tx < K) {
            // Coalesced read from A
            sA[ty][tx] = __ldg(&A[row * K + tile_offset + tx]);
        } else {
            sA[ty][tx] = 0;
        }

        if (tile_offset + ty < K && col < N) {
            // Coalesced read from B
            sB[ty][tx] = __ldg(&B[(tile_offset + ty) * N + col]);
        } else {
            sB[ty][tx] = 0;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    // Coalesced write to C
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);

    TORCH_CHECK(K == B.size(0), "Inner dimensions of matrices must match");

    auto C = torch::empty({M, N}, A.options());

    // Configure grid and block dimensions for better occupancy
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_coalesced_kernel", ([&] {
        matmul_coalesced_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Coalesced matrix multiplication forward (CUDA)");
}