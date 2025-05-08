#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 32
#define ELEMENTS_PER_THREAD 4

template <typename scalar_t>
__global__ void shared_memory_optimized_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L) {

    __shared__ scalar_t sA[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts
    __shared__ scalar_t sB[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int row_base = blockIdx.y * (TILE_DIM * ELEMENTS_PER_THREAD) + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;

    scalar_t thread_results[ELEMENTS_PER_THREAD] = {0};

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        // Collaborative loading of A and B tiles into shared memory
        #pragma unroll
        for (int i = 0; i < TILE_DIM; i += (TILE_DIM * TILE_DIM) / (blockDim.x * blockDim.y)) {
            int load_idx = tid + i;
            if (load_idx < TILE_DIM * TILE_DIM) {
                int load_row = load_idx / TILE_DIM;
                int load_col = load_idx % TILE_DIM;

                // Load elements for matrix A
                #pragma unroll
                for (int e = 0; e < ELEMENTS_PER_THREAD; e++) {
                    int global_row = row_base + e * TILE_DIM;
                    if (global_row < (N * M) && (t * TILE_DIM + load_col) < K) {
                        sA[load_row][load_col] = __ldg(&A[global_row * K + t * TILE_DIM + load_col]);
                    } else {
                        sA[load_row][load_col] = 0;
                    }
                }

                // Load elements for matrix B
                if ((t * TILE_DIM + load_row) < K && (blockIdx.x * TILE_DIM + load_col) < L) {
                    sB[load_row][load_col] = __ldg(&B[(t * TILE_DIM + load_row) * L + blockIdx.x * TILE_DIM + load_col]);
                } else {
                    sB[load_row][load_col] = 0;
                }
            }
        }

        __syncthreads();

        // Compute partial results
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; e++) {
            #pragma unroll
            for (int k = 0; k < TILE_DIM; k++) {
                thread_results[e] += sA[threadIdx.y + e * blockDim.y][k] * sB[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    #pragma unroll
    for (int e = 0; e < ELEMENTS_PER_THREAD; e++) {
        int global_row = row_base + e * TILE_DIM;
        if (global_row < (N * M) && col < L) {
            output[global_row * L + col] = thread_results[e];
        }
    }
}

void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    dim3 threads(TILE_DIM, TILE_DIM/ELEMENTS_PER_THREAD);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, 
              ((N * M) + (TILE_DIM * ELEMENTS_PER_THREAD) - 1) / (TILE_DIM * ELEMENTS_PER_THREAD));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "shared_memory_optimized_kernel", ([&] {
        shared_memory_optimized_kernel<scalar_t><<<grid, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, K, L);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in module_fn_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    auto output = torch::zeros({A.size(0), A.size(1), B.size(1)}, A.options());
    module_fn_cuda_forward(A, B, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Shared memory optimized tensor-matrix multiplication (CUDA)");
}