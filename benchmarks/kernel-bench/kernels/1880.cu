#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    // Calculate global thread ID and warp ID
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    // Each warp handles a row
    const int row = wid;
    
    if (row < N) {
        // Each thread in the warp handles multiple columns
        for (int col_base = lane; col_base < N; col_base += WARP_SIZE) {
            float sum = 0.0f;
            
            if (row >= col_base) {
                // Load data in a coalesced manner
                #pragma unroll 4
                for (int k = col_base; k <= row; k++) {
                    // Consecutive threads read consecutive memory locations
                    const float a_val = A[row * N + k];
                    const float b_val = B[k * N + col_base];
                    sum += a_val * b_val;
                }
                C[row * N + col_base] = sum;
            } else {
                C[row * N + col_base] = 0.0f;
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

    // Calculate grid dimensions based on warps
    const int warps_needed = N;
    const int blocks = (warps_needed * WARP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    triangular_mm_kernel<<<blocks, BLOCK_SIZE>>>(
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