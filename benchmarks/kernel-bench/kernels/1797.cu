#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Declare constant memory for frequently accessed read-only data
__constant__ float A_const[1024 * TILE_SIZE];
__constant__ float B_const[1024 * TILE_SIZE];

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float sum = 0.0f;
    
    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            // Loop over tiles
            for (int t = col/TILE_SIZE; t <= row/TILE_SIZE; t++) {
                // Use constant memory for A and B
                if (row < N && (t*TILE_SIZE + tx) <= row) {
                    As[ty][tx] = A_const[row * N + (t*TILE_SIZE + tx)];
                } else {
                    As[ty][tx] = 0.0f;
                }
                
                if ((t*TILE_SIZE + ty) < N && col < N) {
                    Bs[ty][tx] = B_const[(t*TILE_SIZE + ty) * N + col];
                } else {
                    Bs[ty][tx] = 0.0f;
                }
                
                __syncthreads();
                
                // Compute partial sum for this tile
                if (row < N && col < N && row >= col) {
                    #pragma unroll 8
                    for (int k = 0; k < TILE_SIZE; k++) {
                        if ((t*TILE_SIZE + k) >= col && (t*TILE_SIZE + k) <= row) {
                            sum += As[ty][k] * Bs[k][tx];
                        }
                    }
                }
                
                __syncthreads();
            }
            
            if (row >= col) {
                C[row * N + col] = sum;
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

    // Copy data to constant memory
    cudaMemcpyToSymbol(A_const, A.data_ptr<float>(), A.numel() * sizeof(float));
    cudaMemcpyToSymbol(B_const, B.data_ptr<float>(), B.numel() * sizeof(float));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (N + TILE_SIZE - 1) / TILE_SIZE);

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Launch kernel with stream
    triangular_mm_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular matrix multiplication with constant memory (CUDA)");
}
