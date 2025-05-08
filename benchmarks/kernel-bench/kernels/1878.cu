#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    // Calculate global thread index
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate warp index and lane index
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    // Calculate row and column ensuring coalesced access within warps
    const int row = (warp_id * WARP_SIZE + lane) / N;
    const int col = (warp_id * WARP_SIZE + lane) % N;
    
    if (row < N && col < N) {
        if (row >= col) {  // Lower triangular part
            float sum = 0.0f;
            
            // Ensure coalesced memory access by having consecutive threads
            // access consecutive memory locations
            #pragma unroll 8
            for (int k = col; k <= row; k += 4) {
                // Vector loads for better memory throughput
                float4 a_vec, b_vec;
                if (k + 3 <= row) {
                    // Load 4 consecutive elements
                    a_vec = *reinterpret_cast<const float4*>(&A[row * N + k]);
                    b_vec = *reinterpret_cast<const float4*>(&B[k * N + col]);
                    
                    sum += a_vec.x * b_vec.x;
                    sum += a_vec.y * b_vec.y;
                    sum += a_vec.z * b_vec.z;
                    sum += a_vec.w * b_vec.w;
                } else {
                    // Handle remaining elements
                    for (int kk = k; kk <= row; ++kk) {
                        sum += A[row * N + kk] * B[kk * N + col];
                    }
                    break;
                }
            }
            
            C[row * N + col] = sum;
        } else {
            C[row * N + col] = 0.0f;
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

    // Calculate grid dimensions ensuring proper alignment for coalesced access
    const int num_elements = N * N;
    const int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

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