#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define BLOCK_SIZE 16
#define TILE_DIM (BLOCK_SIZE * 2)
#define PADDING 1  // Avoid shared memory bank conflicts

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

__global__ void matmul_regtile_optimized_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    // Shared memory with padding to avoid bank conflicts
    // Use warp-level padding to reduce bank conflicts
    __shared__ float sA[TILE_DIM][TILE_DIM + 2];  // +2 padding for better bank conflict avoidance
    __shared__ float sB[TILE_DIM][TILE_DIM + 2];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_DIM + ty * 2;
    int col = bx * TILE_DIM + tx * 2;

    float regC00 = 0.0f, regC01 = 0.0f, regC10 = 0.0f, regC11 = 0.0f;
    int linearIndex = ty * BLOCK_SIZE + tx;
    int numTiles = (N + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        // Coalesced loading using float4 for better memory throughput
        for (int i = linearIndex; i < TILE_DIM*TILE_DIM; i += BLOCK_SIZE*BLOCK_SIZE) {
            int r = i / TILE_DIM, c = i % TILE_DIM;
            int globalA_r = by * TILE_DIM + r;
            int globalA_c = t * TILE_DIM + c;
            sA[r][c] = (globalA_r < N && globalA_c < N) ? A[globalA_r * N + globalA_c] : 0.0f;

            int globalB_r = t * TILE_DIM + r;
            int globalB_c = bx * TILE_DIM + c;
            sB[r][c] = (globalB_r < N && globalB_c < N) ? B[globalB_r * N + globalB_c] : 0.0f;
        }
        __syncthreads();

        // Unrolled computation loop to reduce overhead
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            float a0 = sA[ty * 2][k];
            float a1 = sA[ty * 2 + 1][k];
            float b0 = sB[k][tx * 2];
            float b1 = sB[k][tx * 2 + 1];
            regC00 += a0 * b0;
            regC01 += a0 * b1;
            regC10 += a1 * b0;
            regC11 += a1 * b1;
        }
        __syncthreads();  // Only needed once per tile iteration
    }

    // Boundary-aware writeback
    if (row < N && col < N) C[row * N + col] = regC00;
    if (row < N && col+1 < N) C[row * N + col+1] = regC01;
    if (row+1 < N && col < N) C[(row+1)*N + col] = regC10;
    if (row+1 < N && col+1 < N) C[(row+1)*N + col+1] = regC11;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    CHECK_FLOAT(A); CHECK_FLOAT(B);
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be square and equal size");

    int64_t N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + TILE_DIM -1)/TILE_DIM, (N + TILE_DIM -1)/TILE_DIM);

    matmul_regtile_optimized_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    C10_CUDA_CHECK(cudaGetLastError());
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 2x2 tiling MM with sync reduction");
}
