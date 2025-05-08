#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 16

template <typename scalar_t>
__global__ void fully_unrolled_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L) {

    const int row = blockIdx.y * TILE_DIM + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;
    const int batch = blockIdx.z;

    __shared__ scalar_t tile_A[TILE_DIM][TILE_DIM];
    __shared__ scalar_t tile_B[TILE_DIM][TILE_DIM];

    scalar_t sum = 0;

    #pragma unroll
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        if (row < M && (t * TILE_DIM + threadIdx.x) < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[batch * M * K + row * K + t * TILE_DIM + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0;
        }

        if ((t * TILE_DIM + threadIdx.y) < K && col < L) {
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * L + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        scalar_t tmp0  = tile_A[threadIdx.y][0]  * tile_B[0][threadIdx.x];
        scalar_t tmp1  = tile_A[threadIdx.y][1]  * tile_B[1][threadIdx.x];
        scalar_t tmp2  = tile_A[threadIdx.y][2]  * tile_B[2][threadIdx.x];
        scalar_t tmp3  = tile_A[threadIdx.y][3]  * tile_B[3][threadIdx.x];
        scalar_t tmp4  = tile_A[threadIdx.y][4]  * tile_B[4][threadIdx.x];
        scalar_t tmp5  = tile_A[threadIdx.y][5]  * tile_B[5][threadIdx.x];
        scalar_t tmp6  = tile_A[threadIdx.y][6]  * tile_B[6][threadIdx.x];
        scalar_t tmp7  = tile_A[threadIdx.y][7]  * tile_B[7][threadIdx.x];
        scalar_t tmp8  = tile_A[threadIdx.y][8]  * tile_B[8][threadIdx.x];
        scalar_t tmp9  = tile_A[threadIdx.y][9]  * tile_B[9][threadIdx.x];
        scalar_t tmp10 = tile_A[threadIdx.y][10] * tile_B[10][threadIdx.x];
        scalar_t tmp11 = tile_A[threadIdx.y][11] * tile_B[11][threadIdx.x];
        scalar_t tmp12 = tile_A[threadIdx.y][12] * tile_B[12][threadIdx.x];
        scalar_t tmp13 = tile_A[threadIdx.y][13] * tile_B[13][threadIdx.x];
        scalar_t tmp14 = tile_A[threadIdx.y][14] * tile_B[14][threadIdx.x];
        scalar_t tmp15 = tile_A[threadIdx.y][15] * tile_B[15][threadIdx.x];

        sum += tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7 +
               tmp8 + tmp9 + tmp10 + tmp11 + tmp12 + tmp13 + tmp14 + tmp15;

        __syncthreads();
    }

    if (batch < N && row < M && col < L) {
        output[batch * M * L + row * L + col] = sum;
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
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, 
              (M + TILE_DIM - 1) / TILE_DIM,
              N);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "fully_unrolled_kernel", ([&] {
        fully_unrolled_kernel<scalar_t><<<grid, threads>>>(
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
    m.def("forward", &module_fn_forward, "Fully unrolled 3D tensor-matrix multiplication (CUDA)");
}