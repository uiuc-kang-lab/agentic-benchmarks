#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float As[BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE];
    
    // Convert to 1D thread indexing
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    
    // Calculate row and column from linear index
    const int row = idx / N;
    const int col = idx % N;
    
    // Only process elements in the lower triangle
    if (row < N && col < N && row >= col) {
        float sum = 0.0f;
        
        // Process the matrix multiplication in tiles
        for (int t = 0; t < (row - col + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
            const int tileStart = col + t * BLOCK_SIZE;
            
            // Load tile into shared memory
            if (tid < BLOCK_SIZE && tileStart + tid <= row) {
                As[tid] = A[row * N + tileStart + tid];
                Bs[tid] = B[(tileStart + tid) * N + col];
            }
            __syncthreads();
            
            // Compute partial sum for this tile
            const int tileEnd = min(tileStart + BLOCK_SIZE, row + 1);
            for (int k = tileStart; k < tileEnd; ++k) {
                sum += As[k - tileStart] * Bs[k - tileStart];
            }
            
            __syncthreads();
        }
        
        C[row * N + col] = sum;
    } else if (row < N && col < N) {
        C[row * N + col] = 0.0f;
    }
}

// C++ interface exposed to PyTorch
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

    // Calculate grid dimensions for 1D blocks
    const int total_elements = N * N;
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch the CUDA kernel
    triangular_mm_kernel<<<num_blocks, BLOCK_SIZE>>>(
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
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}