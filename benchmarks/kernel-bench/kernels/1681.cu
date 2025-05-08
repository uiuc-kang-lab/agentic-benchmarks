#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    const unsigned int FULL_MASK = 0xffffffff;
    const int warpSize = 32;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = threadIdx.x % warpSize;
    
    if (row >= N || col >= N) return;
    
    // Determine if this warp works on lower triangle
    const int warpRow = row & ~(warpSize-1);
    const int warpCol = col & ~(warpSize-1);
    
    if (warpRow >= warpCol) {
        float sum = 0.0f;
        
        if (row >= col) {
            // Split the work among warp lanes
            const int work_per_thread = (row - col + warpSize) / warpSize;
            const int start_k = col + lane_id;
            
            // Each thread processes its assigned elements
            for (int k_offset = 0; k_offset < work_per_thread; k_offset++) {
                const int k = start_k + k_offset * warpSize;
                if (k <= row) {
                    sum += A[row * N + k] * B[k * N + col];
                }
            }
            
            // Warp-level reduction using shuffle
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(FULL_MASK, sum, offset);
            }
            
            // First lane writes the result
            if (lane_id == 0) {
                C[row * N + col] = sum;
            }
        } else {
            C[row * N + col] = 0.0f;
        }
    } else {
        // Entire warp is in upper triangle
        C[row * N + col] = 0.0f;
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

    // Use 32x32 thread blocks to match warp size
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + 31) / 32, (N + 31) / 32);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Triangular matrix multiplication with warp shuffle (CUDA)");
}