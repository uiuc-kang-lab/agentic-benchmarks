#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 16
#define ALIGN_MASK (~(sizeof(float4) - 1))

template <typename scalar_t>
__global__ void aligned_ldg_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int num_rows,
    const int K,
    const int L) {

    const int row = blockIdx.y * TILE_DIM + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ scalar_t tile_A[TILE_DIM][TILE_DIM + 1]; // +1 for bank conflict avoidance
    __shared__ scalar_t tile_B[TILE_DIM][TILE_DIM + 1];

    scalar_t sum = 0;

    // Align pointers for vector loads where possible
    const scalar_t* aligned_A = reinterpret_cast<const scalar_t*>(
        (reinterpret_cast<uintptr_t>(A + row * K) + 15) & ALIGN_MASK);
    const scalar_t* aligned_B = reinterpret_cast<const scalar_t*>(
        (reinterpret_cast<uintptr_t>(B) + 15) & ALIGN_MASK);

    #pragma unroll 4
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        const int k_offset = t * TILE_DIM;
        
        // Load tile of A using __ldg for read-only cache
        if (row < num_rows && (k_offset + threadIdx.x) < K) {
            tile_A[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + k_offset + threadIdx.x]);
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0;
        }

        // Load tile of B using __ldg for read-only cache
        if ((k_offset + threadIdx.y) < K && col < L) {
            tile_B[threadIdx.y][threadIdx.x] = __ldg(&B[(k_offset + threadIdx.y) * L + col]);
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll 4
        for (int i = 0; i < TILE_DIM; i += 4) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
            sum += tile_A[threadIdx.y][i+1] * tile_B[i+1][threadIdx.x];
            sum += tile_A[threadIdx.y][i+2] * tile_B[i+2][threadIdx.x];
            sum += tile_A[threadIdx.y][i+3] * tile_B[i+3][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result with coalesced access
    if (row < num_rows && col < L) {
        output[row * L + col] = sum;
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
    const int num_rows = N * M;

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, (num_rows + TILE_DIM - 1) / TILE_DIM);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "aligned_ldg_kernel", ([&] {
        aligned_ldg_kernel<scalar_t><<<grid, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_rows, K, L);
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
    m.def("forward", &module_fn_forward, "Aligned LDG tensor matrix multiplication (CUDA)");
}