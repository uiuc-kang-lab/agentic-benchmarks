#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32  // Increased tile size for better occupancy
#define WARP_SIZE 32

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Early exit if above diagonal
    if (row < col || row >= N || col >= N) {
        return;
    }

    float sum = 0.0f;
    
    // Calculate the starting tile for this thread
    const int start_tile = col / TILE_SIZE;
    const int end_tile = row / TILE_SIZE;
    
    #pragma unroll 1
    for (int t = start_tile; t <= end_tile; t++) {
        // Collaborative loading with vectorized memory access
        if (row < N && (t * TILE_SIZE + tx) <= row) {
            As[ty][tx] = __ldg(&A[row * N + (t * TILE_SIZE + tx)]);
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((t * TILE_SIZE + ty) < N && col < N) {
            Bs[ty][tx] = __ldg(&B[(t * TILE_SIZE + ty) * N + col]);
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile using warp-level optimizations
        if ((t * TILE_SIZE) <= row) {
            const int k_start = (t == start_tile) ? col - (t * TILE_SIZE) : 0;
            const int k_end = min(TILE_SIZE, row - (t * TILE_SIZE) + 1);
            
            #pragma unroll 8
            for (int k = k_start; k < k_end; k++) {
                sum = __fmaf_rn(As[ty][k], Bs[k][tx], sum);
            }
        }
        
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    triangular_mm_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular matrix multiplication (CUDA)");
}