#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define BLOCK_COLS 32

template <typename scalar_t>
__global__ void coalesced_memory_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L) {

    __shared__ scalar_t shared_A[BLOCK_ROWS][TILE_DIM];
    __shared__ scalar_t shared_B[TILE_DIM][BLOCK_COLS];

    // Global position
    const int batch = blockIdx.z;
    const int row_block = blockIdx.y * BLOCK_ROWS;
    const int col = blockIdx.x * BLOCK_COLS + threadIdx.x;

    // Local position
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    // Initialize accumulator
    scalar_t acc[BLOCK_ROWS] = {0};

    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_DIM - 1) / TILE_DIM; ++tile) {
        // Collaborative loading of A and B into shared memory
        // Each thread loads multiple elements in a coalesced manner
        #pragma unroll
        for (int i = 0; i < BLOCK_ROWS; i += BLOCK_ROWS/8) {
            int row = row_block + thread_row + i;
            int k = tile * TILE_DIM + thread_col;
            
            if (row < M && k < K && batch < N) {
                shared_A[thread_row + i][thread_col] = A[batch * M * K + row * K + k];
            } else {
                shared_A[thread_row + i][thread_col] = 0;
            }
        }

        // Load B in a coalesced manner
        if (tile * TILE_DIM + thread_row < K && col < L) {
            shared_B[thread_row][thread_col] = B[(tile * TILE_DIM + thread_row) * L + col];
        } else {
            shared_B[thread_row][thread_col] = 0;
        }

        __syncthreads();

        // Compute partial results
        #pragma unroll
        for (int i = 0; i < BLOCK_ROWS; ++i) {
            scalar_t sum = 0;
            #pragma unroll
            for (int k = 0; k < TILE_DIM; k += 4) {
                sum = __fmaf_rn(shared_A[i][k], shared_B[k][thread_col], sum);
                sum = __fmaf_rn(shared_A[i][k+1], shared_B[k+1][thread_col], sum);
                sum = __fmaf_rn(shared_A[i][k+2], shared_B[k+2][thread_col], sum);
                sum = __fmaf_rn(shared_A[i][k+3], shared_B[k+3][thread_col], sum);
            }
            acc[i] += sum;
        }

        __syncthreads();
    }

    // Store results in a coalesced manner
    #pragma unroll
    for (int i = 0; i < BLOCK_ROWS; ++i) {
        int row = row_block + i;
        if (row < M && col < L && batch < N) {
            output[batch * M * L + row * L + col] = acc[i];
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

    dim3 threads(BLOCK_COLS, BLOCK_ROWS/8);
    dim3 grid((L + BLOCK_COLS - 1) / BLOCK_COLS, 
              (M + BLOCK_ROWS - 1) / BLOCK_ROWS,
              N);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "coalesced_memory_kernel", ([&] {
        coalesced_memory_kernel<scalar_t><<<grid, threads>>>(
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
    m.def("forward", &module_fn_forward, "Coalesced memory tensor-matrix multiplication (CUDA)");
}