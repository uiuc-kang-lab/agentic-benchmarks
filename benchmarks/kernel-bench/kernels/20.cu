#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define TILE_SIZE 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

// This kernel partitions the K dimension across the grid's z-dimension.
// Each block computes a partial dot product for a segment of the K dimension
// and then uses atomicAdd to safely accumulate its result to global memory.

__global__ void matmul_k_partition_atomic_kernel(const float* __restrict__ A,
                                                   const float* __restrict__ B,
                                                   float* __restrict__ C,
                                                   int N) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Determine the output matrix indices
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Partition the K dimension using gridDim.z
    int k_start = blockIdx.z * TILE_SIZE;

    float partial = 0.0f;

    // Load the corresponding tile from A and B if within bounds
    if (row < N && col < N) {
        float a_elem, b_elem;

        // Load element from A: row, k_start + tx
        if (k_start + tx < N) {
            a_elem = A[row * N + k_start + tx];
        } else {
            a_elem = 0.0f;
        }

        // Load element from B: k_start + ty, col
        if (k_start + ty < N) {
            b_elem = B[(k_start + ty) * N + col];
        } else {
            b_elem = 0.0f;
        }

        // Each thread loads one element into shared memory
        As[ty][tx] = a_elem;
        Bs[ty][tx] = b_elem;
    } else {
        // If out of bounds, initialize shared memory to zero
        As[ty][tx] = 0.0f;
        Bs[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Compute partial dot product for this tile
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        partial += As[ty][k] * Bs[k][tx];
    }

    // Accumulate the partial result to C using atomicAdd to avoid race conditions
    if (row < N && col < N) {
        atomicAdd(&C[row * N + col], partial);
    }
}


torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);

    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int64_t N = A.size(0);

    // Initialize output tensor C
    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    // Define grid dimensions
    int grid_x = (N + TILE_SIZE - 1) / TILE_SIZE;
    int grid_y = (N + TILE_SIZE - 1) / TILE_SIZE;
    int grid_z = (N + TILE_SIZE - 1) / TILE_SIZE; // Partition K dimension
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(grid_x, grid_y, grid_z);

    // Launch kernel
    matmul_k_partition_atomic_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_data, B_data, C_data, N);
    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication kernel with K-partitioning and atomic accumulation (CUDA)");
}
