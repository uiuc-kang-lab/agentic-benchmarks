#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define THREAD_STRIDE 4

template <typename scalar_t>
__global__ void matmul_cuda_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                                   scalar_t* __restrict__ C, int M, int K, int N) {
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    const int base_row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int base_col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    scalar_t thread_results[THREAD_STRIDE][THREAD_STRIDE] = {0};

    // Loop over tiles
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Collaborative loading of tiles into shared memory
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i += THREAD_STRIDE) {
            if (base_row + i < M && t * TILE_WIDTH + threadIdx.x < K)
                sA[threadIdx.y + i][threadIdx.x] = A[(base_row + i) * K + t * TILE_WIDTH + threadIdx.x];
            if (base_col + i < N && t * TILE_WIDTH + threadIdx.y < K)
                sB[threadIdx.y][threadIdx.x + i] = B[(t * TILE_WIDTH + threadIdx.y) * N + base_col + i];
        }

        __syncthreads();

        // Compute partial results for multiple elements
        #pragma unroll
        for (int i = 0; i < THREAD_STRIDE; ++i) {
            #pragma unroll
            for (int j = 0; j < THREAD_STRIDE; ++j) {
                scalar_t sum = 0;
                #pragma unroll
                for (int k = 0; k < TILE_WIDTH; ++k) {
                    sum += sA[threadIdx.y + i][k] * sB[k][threadIdx.x + j];
                }
                thread_results[i][j] += sum;
            }
        }

        __syncthreads();
    }

    // Write results to global memory with striding
    #pragma unroll
    for (int i = 0; i < THREAD_STRIDE; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_STRIDE; ++j) {
            const int row = base_row + i;
            const int col = base_col + j;
            if (row < M && col < N) {
                C[row * N + col] = thread_results[i][j];
            }
        }
    }
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());

    dim3 threads_per_block(TILE_WIDTH/THREAD_STRIDE, TILE_WIDTH/THREAD_STRIDE);
    dim3 num_blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_cuda_kernel", ([&] {
        matmul_cuda_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication forward (CUDA)");
}