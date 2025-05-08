#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Calculate total number of elements in lower triangle
    const int total_elements = (N * (N + 1)) / 2;
    const int elements_per_block = (total_elements + gridDim.x - 1) / gridDim.x;
    const int start_element = blockIdx.x * elements_per_block;
    const int end_element = min(start_element + elements_per_block, total_elements);
    
    // Each thread processes multiple elements
    for (int idx = start_element + threadIdx.x; idx < end_element; idx += blockDim.x) {
        // Convert 1D index to 2D coordinates in lower triangle
        int row = (int)((-1 + sqrt(1 + 8.0f * idx)) / 2);
        int col = idx - (row * (row + 1)) / 2;
        
        if (row < N && col <= row) {
            float sum = 0.0f;
            
            // Process tiles
            for (int t = 0; t < (row / TILE_SIZE + 1); t++) {
                // Load tiles collaboratively within thread block
                if (threadIdx.x < TILE_SIZE * TILE_SIZE) {
                    int tile_row = threadIdx.x / TILE_SIZE;
                    int tile_col = threadIdx.x % TILE_SIZE;
                    
                    if ((row - tile_row) < N && (t * TILE_SIZE + tile_col) < N) {
                        As[tile_row][tile_col] = A[(row - tile_row) * N + (t * TILE_SIZE + tile_col)];
                    } else {
                        As[tile_row][tile_col] = 0.0f;
                    }
                    
                    if ((t * TILE_SIZE + tile_row) < N && col < N) {
                        Bs[tile_row][tile_col] = B[(t * TILE_SIZE + tile_row) * N + col];
                    } else {
                        Bs[tile_row][tile_col] = 0.0f;
                    }
                }
                
                __syncthreads();
                
                // Compute partial results
                #pragma unroll 8
                for (int k = 0; k < TILE_SIZE; k++) {
                    if ((t * TILE_SIZE + k) <= row) {
                        sum += As[row % TILE_SIZE][k] * Bs[k][col % TILE_SIZE];
                    }
                }
                
                __syncthreads();
            }
            
            C[row * N + col] = sum;
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
    auto C = torch::zeros_like(A);  // Initialize with zeros for upper triangle

    // Calculate optimal number of blocks based on matrix size
    int num_blocks = min(256, (N * N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    triangular_mm_kernel<<<num_blocks, BLOCK_SIZE>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced triangular matrix multiplication (CUDA)");
}