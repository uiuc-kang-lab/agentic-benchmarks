#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    // Calculate global thread ID and warp ID
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid >> 5;  // Divide by 32 (warp size)
    const int lane_id = tid & 31;  // Mod by 32
    const int num_warps = (gridDim.x * blockDim.x) >> 5;
    
    // Each warp processes multiple rows
    for (int row = warp_id; row < N; row += num_warps) {
        // Each thread in the warp processes multiple columns
        for (int col = lane_id; col <= row; col += 32) {
            if (col < N) {  // Ensure we don't go out of bounds
                float sum = 0.0f;
                
                // Process the multiplication in chunks to maximize register usage
                #pragma unroll 4
                for (int k = col; k <= row; k++) {
                    sum += A[row * N + k] * B[k * N + col];
                }
                
                C[row * N + col] = sum;
            }
        }
        
        // Zero out the upper triangular part
        // Each thread handles multiple columns in the upper part
        for (int col = row + 1 + lane_id; col < N; col += 32) {
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

    // Use multiple of warp size for block size
    const int threadsPerBlock = 256;  // 8 warps per block
    const int numBlocks = (N * 32 + threadsPerBlock - 1) / threadsPerBlock;

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
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}