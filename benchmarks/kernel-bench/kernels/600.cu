#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// CUDA kernel for matrix multiplication with manual unrolling of the inner loop
template <typename scalar_t>
__global__ void matmul_unroll_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                                       scalar_t* __restrict__ C, int M, int K, int N) {
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    scalar_t value = 0;

    int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < num_tiles; ++t) {
        int tiledA_col = t * TILE_WIDTH + threadIdx.x;
        int tiledB_row = t * TILE_WIDTH + threadIdx.y;
        
        // Load tile from A
        if (row < M && tiledA_col < K)
            sA[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + tiledA_col]);
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        // Load tile from B
        if (tiledB_row < K && col < N)
            sB[threadIdx.y][threadIdx.x] = __ldg(&B[tiledB_row * N + col]);
        else
            sB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();
        
        // Manually unrolled inner loop since TILE_WIDTH is 16
        value += sA[threadIdx.y][0]  * sB[0][threadIdx.x];
        value += sA[threadIdx.y][1]  * sB[1][threadIdx.x];
        value += sA[threadIdx.y][2]  * sB[2][threadIdx.x];
        value += sA[threadIdx.y][3]  * sB[3][threadIdx.x];
        value += sA[threadIdx.y][4]  * sB[4][threadIdx.x];
        value += sA[threadIdx.y][5]  * sB[5][threadIdx.x];
        value += sA[threadIdx.y][6]  * sB[6][threadIdx.x];
        value += sA[threadIdx.y][7]  * sB[7][threadIdx.x];
        value += sA[threadIdx.y][8]  * sB[8][threadIdx.x];
        value += sA[threadIdx.y][9]  * sB[9][threadIdx.x];
        value += sA[threadIdx.y][10] * sB[10][threadIdx.x];
        value += sA[threadIdx.y][11] * sB[11][threadIdx.x];
        value += sA[threadIdx.y][12] * sB[12][threadIdx.x];
        value += sA[threadIdx.y][13] * sB[13][threadIdx.x];
        value += sA[threadIdx.y][14] * sB[14][threadIdx.x];
        value += sA[threadIdx.y][15] * sB[15][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Forward function accessible from Python
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_unroll_kernel", ([&] {
        matmul_unroll_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication forward with manual loop unrolling (CUDA)");
}
