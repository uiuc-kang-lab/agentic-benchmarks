#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define WARP_SIZE 32
#define TILE_SIZE 32  // Aligned with warp size for better performance

#define CHECK_INPUT(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor"); \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous"); \
    TORCH_CHECK(x.scalar_type() == torch::kFloat, #x " must be a float tensor")

__global__ void warpAlignedMatrixMultiply(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         const int M, const int N, const int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Calculate global indices
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Calculate warp ID and lane ID
    const int warpId = threadIdx.y / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    
    // Compute boundary masks for entire warps to avoid divergent branches
    const bool valid_row = (blockIdx.y * TILE_SIZE + warpId * WARP_SIZE) < M;
    const bool valid_col = (blockIdx.x * TILE_SIZE) < N;
    
    float sum = 0.0f;
    
    // Loop over tiles
    #pragma unroll 1
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load data into shared memory using vectorized loads where possible
        const int tile_offset = t * TILE_SIZE;
        const bool valid_k = (tile_offset + threadIdx.x) < K;
        
        // Entire warp loads or skips together
        if (valid_row && valid_k) {
            As[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + tile_offset + threadIdx.x]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (valid_col && valid_k) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[(tile_offset + threadIdx.y) * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial products - unrolled by 4 for better instruction-level parallelism
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; k += 4) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            sum += As[threadIdx.y][k+1] * Bs[k+1][threadIdx.x];
            sum += As[threadIdx.y][k+2] * Bs[k+2][threadIdx.x];
            sum += As[threadIdx.y][k+3] * Bs[k+3][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write results - entire warp writes or skips together
    if (valid_row && valid_col && row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    // Round up to nearest multiple of TILE_SIZE
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);

    warpAlignedMatrixMultiply<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int M = A.size(0);
    const int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-aligned Matrix Multiplication (CUDA)");
}