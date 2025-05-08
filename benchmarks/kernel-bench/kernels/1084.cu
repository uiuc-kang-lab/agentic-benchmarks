#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 32
#define UNROLL_FACTOR 4

template <typename scalar_t>
__global__ void reduced_sync_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L) {

    const int row = blockIdx.y * TILE_DIM + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;
    const int batch_idx = row / M;
    const int m_idx = row % M;

    __shared__ scalar_t tile_A[TILE_DIM][TILE_DIM];
    __shared__ scalar_t tile_B[TILE_DIM][TILE_DIM];

    scalar_t sum = 0;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        // Load tiles into shared memory
        const int k_offset = t * TILE_DIM + threadIdx.x;
        if (batch_idx < N && m_idx < M && k_offset < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[batch_idx * M * K + m_idx * K + k_offset];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0;
        }

        if ((t * TILE_DIM + threadIdx.y) < K && col < L) {
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * L + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0;
        }

        // Single sync point after loading shared memory
        __syncthreads();

        // Compute partial results
        #pragma unroll
        for (int i = 0; i < TILE_DIM; i += UNROLL_FACTOR) {
            scalar_t a0 = tile_A[threadIdx.y][i];
            scalar_t a1 = tile_A[threadIdx.y][i + 1];
            scalar_t a2 = tile_A[threadIdx.y][i + 2];
            scalar_t a3 = tile_A[threadIdx.y][i + 3];

            scalar_t b0 = tile_B[i][threadIdx.x];
            scalar_t b1 = tile_B[i + 1][threadIdx.x];
            scalar_t b2 = tile_B[i + 2][threadIdx.x];
            scalar_t b3 = tile_B[i + 3][threadIdx.x];

            sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        }

        // Single sync point before next iteration
        __syncthreads();
    }

    // Write result
    if (batch_idx < N && m_idx < M && col < L) {
        output[batch_idx * M * L + m_idx * L + col] = sum;
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

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, (N * M + TILE_DIM - 1) / TILE_DIM);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "reduced_sync_kernel", ([&] {
        reduced_sync_kernel<scalar_t><<<grid, threads>>>(
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

    auto N = A.size(0);
    auto M = A.size(1);
    auto L = B.size(1);

    auto output = torch::zeros({N, M, L}, A.options());
    module_fn_cuda_forward(A, B, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Reduced sync tensor-matrix multiplication (CUDA)");
}