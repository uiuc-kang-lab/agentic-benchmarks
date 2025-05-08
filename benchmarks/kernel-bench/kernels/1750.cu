#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 128
#define TILE_DIM 16

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float As[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts
    __shared__ float Bs[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    const int row = blockIdx.y * TILE_DIM + (tid / TILE_DIM);
    const int col = blockIdx.x * TILE_DIM + (tid % TILE_DIM);

    float sum = 0.0f;

    // Only compute for lower triangular portion
    if (row < N && col < N && row >= col) {
        for (int t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; ++t) {
            const int tile_row = row;
            const int tile_col = t * TILE_DIM + lane % TILE_DIM;

            // Collaborative loading into shared memory
            if (tid < TILE_DIM * TILE_DIM) {
                const int sm_row = tid / TILE_DIM;
                const int sm_col = tid % TILE_DIM;
                
                if (tile_row < N && t * TILE_DIM + sm_col < N && tile_row >= t * TILE_DIM + sm_col) {
                    As[sm_row][sm_col] = A[tile_row * N + (t * TILE_DIM + sm_col)];
                } else {
                    As[sm_row][sm_col] = 0.0f;
                }

                if (t * TILE_DIM + sm_row < N && col < N && col <= t * TILE_DIM + sm_row) {
                    Bs[sm_row][sm_col] = B[(t * TILE_DIM + sm_row) * N + col];
                } else {
                    Bs[sm_row][sm_col] = 0.0f;
                }
            }
            
            __syncthreads();

            // Compute partial results
            if (row < N && col < N && row >= col) {
                #pragma unroll
                for (int k = 0; k < TILE_DIM; ++k) {
                    const int global_k = t * TILE_DIM + k;
                    if (global_k >= col && global_k <= row) {
                        sum += As[tid / TILE_DIM][k] * Bs[k][tid % TILE_DIM];
                    }
                }
            }
            
            __syncthreads();
        }

        // Write result
        if (row < N && col < N) {
            if (row >= col) {
                C[row * N + col] = sum;
            } else {
                C[row * N + col] = 0.0f;
            }
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    triangular_mm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication with warp-optimized shared memory (CUDA)");
}