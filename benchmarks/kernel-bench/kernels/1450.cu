#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__constant__ int d_N;

__global__ void matmul_kernel_warp_scheduled(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    const int warpSize = 32;
    const int warpId = threadIdx.x / warpSize;
    const int laneId = threadIdx.x % warpSize;

    float value = 0.0f;

    const int blockCol = blockIdx.x * TILE_SIZE;
    const int blockRow = blockIdx.y * TILE_SIZE;

    for (int t = 0; t < (d_N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Efficient loading into shared memory partitioned by warp
        int row = blockRow + warpId;
        int col = blockCol + laneId;

        if (row < d_N && (t * TILE_SIZE + laneId) < d_N) {
            s_A[warpId][laneId] = A[row * d_N + (t * TILE_SIZE + laneId)];
        }
        if ((t * TILE_SIZE + warpId) < d_N && col < d_N) {
            s_B[warpId][laneId] = B[(t * TILE_SIZE + warpId) * d_N + col];
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            if ((t * TILE_SIZE + k) < d_N) {
                value += s_A[warpId][k] * s_B[k][laneId];
            }
        }
        __syncthreads();
    }

    row = blockRow + warpId;
    col = blockCol + laneId;
    if (row < d_N && col < d_N) {
        C[row * d_N + col] = value;
    }
}

// C++ interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of same size");

    int N = A.size(0);
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    dim3 threads(TILE_SIZE, TILE_SIZE / 2); // Utilize warp-level parallelism
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel_warp_scheduled<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Matrix Multiplication (CUDA)");
}
