#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    // Use 2D blocks but with careful thread assignment
    const int warp_size = 32;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int bid = blockIdx.x + blockIdx.y * gridDim.x;
    const int num_threads = blockDim.x * blockDim.y;
    const int thread_id = bid * num_threads + tid;
    
    // Each thread processes one row
    const int row = thread_id;
    
    if (row < N) {
        // Process entire row in the lower triangular part
        for (int col = 0; col <= row; col++) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int k = col; k <= row; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
        
        // Zero out the upper triangular part
        for (int col = row + 1; col < N; col++) {
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

    // Use 32x8 thread blocks for better occupancy
    dim3 threadsPerBlock(32, 8);
    dim3 numBlocks((N + 255) / 256, 1);

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