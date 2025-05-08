#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define WARP_SIZE 32

__global__ void matmul_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int N) {
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float value = 0.0f;
    
    // Each thread processes one row-column pair
    if (row < N && col < N) {
        for (int k = 0; k < N; k += WARP_SIZE) {
            // Load data into registers
            float a_reg = (k + lane_id < N) ? A[row * N + k + lane_id] : 0.0f;
            float b_reg = (k + lane_id < N) ? B[(k + lane_id) * N + col] : 0.0f;
            
            // Perform warp-level dot product
            #pragma unroll
            for (int offset = 0; offset < WARP_SIZE; ++offset) {
                // Broadcast a_reg and b_reg within the warp
                float a_val = __shfl_sync(0xffffffff, a_reg, offset);
                float b_val = __shfl_sync(0xffffffff, b_reg, offset);
                value = fmaf(a_val, b_val, value);
            }
        }
        
        // Write result
        C[row * N + col] = value;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of same size");

    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp Shuffle Matrix Multiplication (CUDA)");
}