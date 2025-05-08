#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 32
#define THREAD_TILE 2
#define NUM_STREAMS 4
#define CHUNK_SIZE 1024  // Size of matrix chunks to be processed by each stream

__constant__ int d_N;

__global__ void matmul_kernel_stream(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     const int chunk_start,
                                     const int chunk_size) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    // Global row and column indices, offset by chunk_start
    int row = chunk_start + blockIdx.y * BLOCK_SIZE + ty * THREAD_TILE;
    int col = blockIdx.x * BLOCK_SIZE + tx * THREAD_TILE;

    float regC[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    // Number of tiles for this chunk
    int num_tiles = (d_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Load tiles from A and B into shared memory
        for (int i = 0; i < THREAD_TILE; i++) {
            for (int j = 0; j < THREAD_TILE; j++) {
                int global_row = row + i;
                int global_col = t * BLOCK_SIZE + tx * THREAD_TILE + j;
                
                if (global_row < d_N && global_col < d_N) {
                    s_A[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 
                        __ldg(&A[global_row * d_N + global_col]);
                } else {
                    s_A[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 0.0f;
                }

                global_row = t * BLOCK_SIZE + ty * THREAD_TILE + i;
                global_col = col + j;
                
                if (global_row < d_N && global_col < d_N) {
                    s_B[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 
                        __ldg(&B[global_row * d_N + global_col]);
                } else {
                    s_B[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            for (int i = 0; i < THREAD_TILE; i++) {
                float a_val = s_A[ty * THREAD_TILE + i][k];
                for (int j = 0; j < THREAD_TILE; j++) {
                    regC[i][j] += a_val * s_B[k][tx * THREAD_TILE + j];
                }
            }
        }

        __syncthreads();
    }

    // Store results
    for (int i = 0; i < THREAD_TILE; i++) {
        for (int j = 0; j < THREAD_TILE; j++) {
            int global_row = row + i;
            int global_col = col + j;
            if (global_row < chunk_start + chunk_size && global_row < d_N && global_col < d_N) {
                C[global_row * d_N + global_col] = regC[i][j];
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
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same dimensions");

    int N = A.size(0);
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 threads(BLOCK_SIZE / THREAD_TILE, BLOCK_SIZE / THREAD_TILE);
    
    for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE) {
        int chunk_size = std::min(CHUNK_SIZE, N - chunk_start);
        int num_blocks_y = (chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int num_blocks_x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 blocks(num_blocks_x, num_blocks_y);
        
        int stream_idx = (chunk_start / CHUNK_SIZE) % NUM_STREAMS;
        
        matmul_kernel_stream<<<blocks, threads, 0, streams[stream_idx]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            chunk_start,
            chunk_size
        );
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stream-based Matrix Multiplication (CUDA)");
}