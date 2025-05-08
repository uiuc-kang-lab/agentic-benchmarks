#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Tile size used for shared memory tiling
#define TILE_SIZE 16

// Optimized kernel using shared memory tiling with __ldg() for read-only accesses
__global__ void matrix_mul_ldg_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Optimized loop with double buffering to overlap shared memory loads with computation
    __shared__ float tile_A_db[2][TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B_db[2][TILE_SIZE][TILE_SIZE];

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    int curr = 0, next = 1;

    // Preload the first tile into buffer 'curr'
    int t0 = 0;
    int tiledX = t0 * TILE_SIZE + threadIdx.x;
    int tiledY = t0 * TILE_SIZE + threadIdx.y;
    if (row < M && tiledX < K)
        tile_A_db[curr][threadIdx.y][threadIdx.x] = __ldg(&A[row * K + tiledX]);
    else
        tile_A_db[curr][threadIdx.y][threadIdx.x] = 0.0f;
    if (tiledY < K && col < N)
        tile_B_db[curr][threadIdx.y][threadIdx.x] = __ldg(&B[tiledY * N + col]);
    else
        tile_B_db[curr][threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // Loop over tiles with double buffering
    for (int t = 0; t < numTiles - 1; t++) {
        int t_next = t + 1;
        // Asynchronously preload the next tile into buffer 'next'
        int tiledX_next = t_next * TILE_SIZE + threadIdx.x;
        int tiledY_next = t_next * TILE_SIZE + threadIdx.y;
        if (row < M && tiledX_next < K)
            tile_A_db[next][threadIdx.y][threadIdx.x] = __ldg(&A[row * K + tiledX_next]);
        else
            tile_A_db[next][threadIdx.y][threadIdx.x] = 0.0f;
        if (tiledY_next < K && col < N)
            tile_B_db[next][threadIdx.y][threadIdx.x] = __ldg(&B[tiledY_next * N + col]);
        else
            tile_B_db[next][threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial multiplication using the current tile in buffer 'curr'
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A_db[curr][threadIdx.y][k] * tile_B_db[curr][k][threadIdx.x];
        }

        // Swap the buffers
        curr = next;
        next = 1 - curr;

        __syncthreads();
    }

    // Process the last tile
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += tile_A_db[curr][threadIdx.y][k] * tile_B_db[curr][k][threadIdx.x];
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Launcher function to execute the kernel
void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matrix_mul_ldg_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
}

// Pybind interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with read-only __ldg() and aligned loads (CUDA)");
}
