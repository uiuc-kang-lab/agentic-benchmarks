#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define SUBTILE 4  // For manual unrolling of inner loops

__global__ void unrolled_triangular_mm_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               const int N) {
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Early exit for out-of-bounds threads
    if (row >= N || col >= N) return;

    // Block-level early exit for upper triangular region
    const int block_row_max = blockIdx.y * TILE_SIZE + TILE_SIZE - 1;
    const int block_col_min = blockIdx.x * TILE_SIZE;
    if (block_row_max < block_col_min) {
        C[row * N + col] = 0.0f;
        return;
    }

    // Shared memory tiles
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // Register cache for partial sums to reduce shared memory pressure
    float sum[SUBTILE] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    #pragma unroll 1  // Allow compiler to determine optimal unrolling for outer loop
    for (int m = 0; m < numTiles; m++) {
        // Load tiles with unrolled accesses
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i += SUBTILE) {
            const int k_A = m * TILE_SIZE + threadIdx.x + i;
            const int k_B = m * TILE_SIZE + threadIdx.y + i;
            
            if (row < N && k_A < N && row >= k_A) {
                sA[threadIdx.y][threadIdx.x + i] = A[row * N + k_A];
                sA[threadIdx.y][threadIdx.x + i + 1] = A[row * N + k_A + 1];
                sA[threadIdx.y][threadIdx.x + i + 2] = A[row * N + k_A + 2];
                sA[threadIdx.y][threadIdx.x + i + 3] = A[row * N + k_A + 3];
            } else {
                sA[threadIdx.y][threadIdx.x + i] = 0.0f;
                sA[threadIdx.y][threadIdx.x + i + 1] = 0.0f;
                sA[threadIdx.y][threadIdx.x + i + 2] = 0.0f;
                sA[threadIdx.y][threadIdx.x + i + 3] = 0.0f;
            }

            if (k_B < N && col < N && k_B >= col) {
                sB[threadIdx.y + i][threadIdx.x] = B[k_B * N + col];
                sB[threadIdx.y + i + 1][threadIdx.x] = B[(k_B + 1) * N + col];
                sB[threadIdx.y + i + 2][threadIdx.x] = B[(k_B + 2) * N + col];
                sB[threadIdx.y + i + 3][threadIdx.x] = B[(k_B + 3) * N + col];
            } else {
                sB[threadIdx.y + i][threadIdx.x] = 0.0f;
                sB[threadIdx.y + i + 1][threadIdx.x] = 0.0f;
                sB[threadIdx.y + i + 2][threadIdx.x] = 0.0f;
                sB[threadIdx.y + i + 3][threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();

        // Compute valid k range for this tile
        const int tile_start = m * TILE_SIZE;
        const int k_start = max(col, tile_start);
        const int k_end = min(row + 1, min(N, tile_start + TILE_SIZE));
        
        if (k_start < k_end) {
            // Process tile with aggressive unrolling
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k += SUBTILE) {
                if (tile_start + k >= k_start && tile_start + k < k_end) {
                    #pragma unroll
                    for (int s = 0; s < SUBTILE; s++) {
                        if (tile_start + k + s < k_end) {
                            const float a_val = sA[threadIdx.y][k + s];
                            const float b_val = sB[k + s][threadIdx.x];
                            sum[s] += a_val * b_val;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Combine partial sums
    float final_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < SUBTILE; i++) {
        final_sum += sum[i];
    }

    // Write result, ensuring zero for upper triangular region
    C[row * N + col] = (row < col) ? 0.0f : final_sum;
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

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    unrolled_triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Unrolled Triangular Matrix Multiplication (CUDA)");
}