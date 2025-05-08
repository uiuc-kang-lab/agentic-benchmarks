#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define STRIDE 4

__global__ void strided_triangular_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    
    // Base indices for the block
    const int blockRow = blockIdx.y * TILE_SIZE;
    const int blockCol = blockIdx.x * TILE_SIZE;

    // Process multiple elements per thread using strides
    #pragma unroll
    for (int s = 0; s < STRIDE; s++) {
        const int row = blockRow + (tid + s * blockDim.x * blockDim.y) / TILE_SIZE;
        const int col = blockCol + (tid + s * blockDim.x * blockDim.y) % TILE_SIZE;

        float sum = 0.0f;

        // Tile multiplication
        for (int t = 0; t <= (row / TILE_SIZE); t++) {
            // Collaborative loading of tiles into shared memory
            if (tid < TILE_SIZE * TILE_SIZE) {
                int loadRow = blockRow + tid / TILE_SIZE;
                int loadCol = t * TILE_SIZE + tid % TILE_SIZE;
                As[tid / TILE_SIZE][tid % TILE_SIZE] = (loadRow < N && loadCol < N && loadRow >= loadCol) 
                    ? __ldg(&A[loadRow * N + loadCol]) : 0.0f;

                loadRow = t * TILE_SIZE + tid / TILE_SIZE;
                loadCol = blockCol + tid % TILE_SIZE;
                Bs[tid / TILE_SIZE][tid % TILE_SIZE] = (loadRow < N && loadCol < N && loadRow >= loadCol) 
                    ? __ldg(&B[loadRow * N + loadCol]) : 0.0f;
            }
            
            __syncthreads();

            if (row < N && col < N) {
                #pragma unroll
                for (int k = 0; k < TILE_SIZE; k++) {
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }
            
            __syncthreads();
        }

        // Write result
        if (row < N && col < N) {
            C[row * N + col] = (row >= col) ? sum : 0.0f;
        }
    }
}

at::Tensor forward(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be same size");

    const int N = static_cast<int>(A.size(0));
    auto C = torch::empty_like(A);

    // Adjust grid size based on stride
    dim3 block(TILE_SIZE, TILE_SIZE / STRIDE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    strided_triangular_mm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided triangular matrix multiplication (CUDA)");
}