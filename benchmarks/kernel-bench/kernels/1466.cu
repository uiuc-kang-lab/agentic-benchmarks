#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16  // Reduced block size for better occupancy
#define THREAD_TILE 2
#define CHUNK_SIZE 4
#define MAX_MATRIX_DIM 8192

__constant__ int d_N;

__global__ void matmul_kernel_distributed(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    // Calculate initial position using block-cyclic distribution
    const int num_blocks_x = (d_N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int block_id = blockIdx.y * num_blocks_x + blockIdx.x;
    const int total_blocks = gridDim.x * gridDim.y;
    
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Initialize accumulator registers
    float acc[THREAD_TILE][THREAD_TILE] = {0.0f};
    
    // Process multiple chunks cyclically
    for (int chunk = block_id; chunk < (d_N * d_N) / (BLOCK_SIZE * BLOCK_SIZE); chunk += total_blocks) {
        // Convert chunk index back to 2D coordinates
        const int chunk_row = (chunk / num_blocks_x) * BLOCK_SIZE;
        const int chunk_col = (chunk % num_blocks_x) * BLOCK_SIZE;

        // Process CHUNK_SIZE tiles per chunk to improve data reuse
        for (int t = 0; t < CHUNK_SIZE && (chunk_row + t * BLOCK_SIZE) < d_N; t++) {
            const int current_row = chunk_row + t * BLOCK_SIZE;
            
            // Load tile from A using __ldg
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                for (int j = 0; j < THREAD_TILE; j++) {
                    const int row = current_row + ty * THREAD_TILE + i;
                    const int col = chunk_col + tx * THREAD_TILE + j;
                    if (row < d_N && col < d_N) {
                        s_A[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = __ldg(&A[row * d_N + col]);
                    } else {
                        s_A[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 0.0f;
                    }
                }
            }

            // Load tile from B using __ldg
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                for (int j = 0; j < THREAD_TILE; j++) {
                    const int row = chunk_col + ty * THREAD_TILE + i;
                    const int col = current_row + tx * THREAD_TILE + j;
                    if (row < d_N && col < d_N) {
                        s_B[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = __ldg(&B[row * d_N + col]);
                    } else {
                        s_B[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 0.0f;
                    }
                }
            }

            __syncthreads();

            // Compute matrix multiplication for current tile
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; k++) {
                #pragma unroll
                for (int i = 0; i < THREAD_TILE; i++) {
                    for (int j = 0; j < THREAD_TILE; j++) {
                        acc[i][j] += s_A[ty * THREAD_TILE + i][k] * s_B[k][tx * THREAD_TILE + j];
                    }
                }
            }

            __syncthreads();
        }

        // Write results back to global memory
        const int out_row = chunk_row + ty * THREAD_TILE;
        const int out_col = chunk_col + tx * THREAD_TILE;

        #pragma unroll
        for (int i = 0; i < THREAD_TILE; i++) {
            for (int j = 0; j < THREAD_TILE; j++) {
                if (out_row + i < d_N && out_col + j < d_N) {
                    C[(out_row + i) * d_N + out_col + j] = acc[i][j];
                }
            }
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have same size");
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds maximum supported size");

    int N = A.size(0);
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Calculate grid dimensions for better work distribution
    dim3 threads(BLOCK_SIZE / THREAD_TILE, BLOCK_SIZE / THREAD_TILE);
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(num_blocks, num_blocks);

    matmul_kernel_distributed<<<blocks, threads>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>()
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Distributed Matrix Multiplication (CUDA)");
}