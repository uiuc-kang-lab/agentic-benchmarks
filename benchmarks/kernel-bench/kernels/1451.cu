#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

__global__ void matmul_kernel_warp(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  const int N) {
    // Shared memory for the tiles
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    // Thread and warp identification
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    // Global row and column
    const int row = blockIdx.y * TILE_SIZE + wid * (TILE_SIZE/WARPS_PER_BLOCK) + lane / (TILE_SIZE/WARPS_PER_BLOCK);
    const int col = blockIdx.x * TILE_SIZE + lane % TILE_SIZE;

    // Accumulator registers
    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Collaborative loading of tiles into shared memory
        const int loadRow = wid * (TILE_SIZE/WARPS_PER_BLOCK) + lane / (TILE_SIZE/WARPS_PER_BLOCK);
        const int loadCol = lane % TILE_SIZE;
        
        if (blockIdx.y * TILE_SIZE + loadRow < N && t * TILE_SIZE + loadCol < N) {
            s_A[loadRow][loadCol] = A[(blockIdx.y * TILE_SIZE + loadRow) * N + t * TILE_SIZE + loadCol];
        } else {
            s_A[loadRow][loadCol] = 0.0f;
        }

        if (t * TILE_SIZE + loadRow < N && blockIdx.x * TILE_SIZE + loadCol < N) {
            s_B[loadRow][loadCol] = B[(t * TILE_SIZE + loadRow) * N + blockIdx.x * TILE_SIZE + loadCol];
        } else {
            s_B[loadRow][loadCol] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot products
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[wid * (TILE_SIZE/WARPS_PER_BLOCK) + lane / (TILE_SIZE/WARPS_PER_BLOCK)][k] * 
                   s_B[k][lane % TILE_SIZE];
        }

        __syncthreads();
    }

    // Write results to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have matching dimensions");

    const int N = A.size(0);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Launch configuration
    dim3 threads(THREADS_PER_BLOCK);  // 256 threads per block (8 warps)
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel_warp<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized Matrix Multiplication (CUDA)");
}