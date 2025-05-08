#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define ELEMENTS_PER_THREAD 4

__global__ void balanced_workload_tril_mm_kernel(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float* __restrict__ C,
                                                const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;
    
    // Calculate base indices for this block
    const int block_row = blockIdx.y * TILE_SIZE;
    const int block_col = blockIdx.x * TILE_SIZE;

    // Pre-calculate thread's assigned rows and columns
    const int thread_rows[ELEMENTS_PER_THREAD] = {
        block_row + (tid / TILE_SIZE) * ELEMENTS_PER_THREAD,
        block_row + (tid / TILE_SIZE) * ELEMENTS_PER_THREAD + 1,
        block_row + (tid / TILE_SIZE) * ELEMENTS_PER_THREAD + 2,
        block_row + (tid / TILE_SIZE) * ELEMENTS_PER_THREAD + 3
    };
    
    const int thread_cols[ELEMENTS_PER_THREAD] = {
        block_col + (tid % TILE_SIZE),
        block_col + (tid % TILE_SIZE) + TILE_SIZE,
        block_col + (tid % TILE_SIZE) + 2 * TILE_SIZE,
        block_col + (tid % TILE_SIZE) + 3 * TILE_SIZE
    };

    float sums[ELEMENTS_PER_THREAD] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Process tiles
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Cooperative loading of tiles into shared memory
        const int tile_offset = tile * TILE_SIZE;
        
        // Load multiple elements per thread
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i += blockDim.x * blockDim.y / TILE_SIZE) {
            int load_idx = tid + i;
            if (load_idx < TILE_SIZE * TILE_SIZE) {
                int row = load_idx / TILE_SIZE;
                int col = load_idx % TILE_SIZE;
                
                if (block_row + row < N && tile_offset + col < N) {
                    As[row][col] = A[(block_row + row) * N + tile_offset + col];
                    Bs[row][col] = B[(tile_offset + row) * N + block_col + col];
                } else {
                    As[row][col] = 0.0f;
                    Bs[row][col] = 0.0f;
                }
            }
        }
        
        __syncthreads();

        // Compute partial results for multiple elements
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            if (thread_rows[e] < N && thread_cols[e] < N && thread_rows[e] >= thread_cols[e]) {
                #pragma unroll
                for (int k = 0; k < TILE_SIZE; ++k) {
                    if (tile_offset + k >= thread_cols[e] && tile_offset + k <= thread_rows[e]) {
                        sums[e] += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                    }
                }
            }
        }
        
        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
        if (thread_rows[e] < N && thread_cols[e] < N) {
            if (thread_rows[e] >= thread_cols[e]) {
                C[thread_rows[e] * N + thread_cols[e]] = sums[e];
            } else {
                C[thread_rows[e] * N + thread_cols[e]] = 0.0f;
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

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE/ELEMENTS_PER_THREAD);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    balanced_workload_tril_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Balanced workload triangular matrix multiplication (CUDA)");
}