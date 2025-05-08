#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define THREAD_TILE 4
#define PAD 1  // Padding to avoid shared memory bank conflicts
#define MAX_MATRIX_DIM 8192

__constant__ int d_N;
__constant__ int d_num_tiles;

__global__ void shared_vec_ldg_matmul(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C) {
    // Padded shared memory to avoid bank conflicts
    __shared__ float s_A[2][BLOCK_SIZE][BLOCK_SIZE + PAD];
    __shared__ float s_B[2][BLOCK_SIZE][BLOCK_SIZE + PAD];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Global memory indices
    const int row_start = by * BLOCK_SIZE + ty * THREAD_TILE;
    const int col_start = bx * BLOCK_SIZE + tx * THREAD_TILE;

    // Accumulation registers
    float reg_C[THREAD_TILE][THREAD_TILE] = {0.0f};
    
    // Double buffering indices
    int load_buffer = 0;
    int compute_buffer = 0;

    // Prefetch first tile
    {
        const int threads_per_block = blockDim.x * blockDim.y;
        const int thread_id = ty * blockDim.x + tx;
        
        // Each thread loads multiple elements using float4
        for (int i = thread_id; i < (BLOCK_SIZE * BLOCK_SIZE) / 4; i += threads_per_block) {
            int row = i / (BLOCK_SIZE/4);
            int col = (i % (BLOCK_SIZE/4)) * 4;
            
            // Load A tile
            float4 a_vec;
            int global_row_a = by * BLOCK_SIZE + row;
            int global_col_a = col;
            if (global_row_a < d_N && global_col_a + 3 < d_N) {
                a_vec = *reinterpret_cast<const float4*>(&A[global_row_a * d_N + global_col_a]);
            } else {
                a_vec.x = a_vec.y = a_vec.z = a_vec.w = 0.0f;
            }
            
            s_A[load_buffer][row][col] = a_vec.x;
            s_A[load_buffer][row][col + 1] = a_vec.y;
            s_A[load_buffer][row][col + 2] = a_vec.z;
            s_A[load_buffer][row][col + 3] = a_vec.w;

            // Load B tile
            float4 b_vec;
            int global_row_b = row;
            int global_col_b = bx * BLOCK_SIZE + col;
            if (global_row_b < d_N && global_col_b + 3 < d_N) {
                b_vec = *reinterpret_cast<const float4*>(&B[global_row_b * d_N + global_col_b]);
            } else {
                b_vec.x = b_vec.y = b_vec.z = b_vec.w = 0.0f;
            }
            
            s_B[load_buffer][row][col] = b_vec.x;
            s_B[load_buffer][row][col + 1] = b_vec.y;
            s_B[load_buffer][row][col + 2] = b_vec.z;
            s_B[load_buffer][row][col + 3] = b_vec.w;
        }
    }
    __syncthreads();

    // Main loop
    #pragma unroll 1
    for (int tile = 0; tile < d_num_tiles; ++tile) {
        // Load next tile while computing current tile
        if (tile + 1 < d_num_tiles) {
            const int next_tile = tile + 1;
            const int threads_per_block = blockDim.x * blockDim.y;
            const int thread_id = ty * blockDim.x + tx;
            
            load_buffer = 1 - compute_buffer;
            
            for (int i = thread_id; i < (BLOCK_SIZE * BLOCK_SIZE) / 4; i += threads_per_block) {
                int row = i / (BLOCK_SIZE/4);
                int col = (i % (BLOCK_SIZE/4)) * 4;
                
                // Load next A tile
                float4 a_vec;
                int global_row_a = by * BLOCK_SIZE + row;
                int global_col_a = next_tile * BLOCK_SIZE + col;
                if (global_row_a < d_N && global_col_a + 3 < d_N) {
                    a_vec = *reinterpret_cast<const float4*>(&A[global_row_a * d_N + global_col_a]);
                } else {
                    a_vec.x = a_vec.y = a_vec.z = a_vec.w = 0.0f;
                }
                
                s_A[load_buffer][row][col] = a_vec.x;
                s_A[load_buffer][row][col + 1] = a_vec.y;
                s_A[load_buffer][row][col + 2] = a_vec.z;
                s_A[load_buffer][row][col + 3] = a_vec.w;

                // Load next B tile
                float4 b_vec;
                int global_row_b = next_tile * BLOCK_SIZE + row;
                int global_col_b = bx * BLOCK_SIZE + col;
                if (global_row_b < d_N && global_col_b + 3 < d_N) {
                    b_vec = *reinterpret_cast<const float4*>(&B[global_row_b * d_N + global_col_b]);
                } else {
                    b_vec.x = b_vec.y = b_vec.z = b_vec.w = 0.0f;
                }
                
                s_B[load_buffer][row][col] = b_vec.x;
                s_B[load_buffer][row][col + 1] = b_vec.y;
                s_B[load_buffer][row][col + 2] = b_vec.z;
                s_B[load_buffer][row][col + 3] = b_vec.w;
            }
        }

        // Compute current tile
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            // Load values from shared memory
            float a_reg[THREAD_TILE];
            float b_reg[THREAD_TILE];
            
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; ++i) {
                a_reg[i] = s_A[compute_buffer][ty * THREAD_TILE + i][k];
                b_reg[i] = s_B[compute_buffer][k][tx * THREAD_TILE + i];
            }

            // Compute outer product
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE; ++j) {
                    reg_C[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        compute_buffer = 1 - compute_buffer;
        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; ++i) {
        const int global_row = row_start + i;
        if (global_row < d_N) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; ++j) {
                const int global_col = col_start + j;
                if (global_col < d_N) {
                    C[global_row * d_N + global_col] = reg_C[i][j];
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
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix too large");

    int N = A.size(0);
    int num_tiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMemcpyToSymbol(d_N, &N, sizeof(int));
    cudaMemcpyToSymbol(d_num_tiles, &num_tiles, sizeof(int));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    dim3 threads(BLOCK_SIZE / THREAD_TILE, BLOCK_SIZE / THREAD_TILE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    shared_vec_ldg_matmul<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared Memory Optimized Matrix Multiplication (CUDA)");
}