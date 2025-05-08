#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 128;  // Optimized for H100
constexpr int VECTOR_SIZE = 4;   // Use vectorized loads where possible

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int total_threads = gridDim.x * BLOCK_SIZE;
    
    // Each thread processes multiple elements for better utilization
    for (int idx = tid; idx < N * N; idx += total_threads) {
        const int row = idx / N;
        const int col = idx % N;
        
        if (row < N && col < N) {
            if (row < col) {
                C[idx] = 0.0f;
            } else {
                float sum = 0.0f;
                
                // Process elements in vectors of 4 where possible
                const int aligned_start = (col + VECTOR_SIZE - 1) & ~(VECTOR_SIZE - 1);
                const int aligned_end = (row + 1) & ~(VECTOR_SIZE - 1);
                
                // Handle initial unaligned elements
                for (int k = col; k < aligned_start && k <= row; ++k) {
                    sum += A[row * N + k] * B[k * N + col];
                }
                
                // Vectorized main loop
                #pragma unroll 4
                for (int k = aligned_start; k < aligned_end; k += VECTOR_SIZE) {
                    float4 a_vec = *reinterpret_cast<const float4*>(&A[row * N + k]);
                    float4 b_vec = *reinterpret_cast<const float4*>(&B[k * N + col]);
                    sum += a_vec.x * b_vec.x + a_vec.y * b_vec.y + 
                          a_vec.z * b_vec.z + a_vec.w * b_vec.w;
                }
                
                // Handle remaining elements
                for (int k = aligned_end; k <= row; ++k) {
                    sum += A[row * N + k] * B[k * N + col];
                }
                
                C[idx] = sum;
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

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // Calculate optimal grid size based on block size
    const int num_elements = N * N;
    const int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Limit number of blocks to maintain good occupancy
    const int max_blocks = 32768;  // Typical maximum for H100
    const int grid_size = min(num_blocks, max_blocks);

    triangular_mm_kernel<<<grid_size, BLOCK_SIZE>>>(
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