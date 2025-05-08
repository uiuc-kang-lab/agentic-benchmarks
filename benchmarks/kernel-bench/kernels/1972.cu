#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void grid_strided_triangular_mm(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    // Grid-strided loop setup
    int tile_start_row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int tile_start_col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Calculate grid-strided loop increments
    int row_stride = gridDim.y * TILE_SIZE;
    int col_stride = gridDim.x * TILE_SIZE;

    for (int row_base = tile_start_row; row_base < N; row_base += row_stride) {
        for (int col_base = tile_start_col; col_base <= row_base; col_base += col_stride) {
            int row = row_base + threadIdx.y;
            int col = col_base + threadIdx.x;
            
            if (row >= N || col >= N || row < col) continue;

            float sum = 0.0f;
            int last_valid_tile = (row + TILE_SIZE - 1) / TILE_SIZE;

            for (int m = 0; m < last_valid_tile; ++m) {
                // Load A tile
                int load_col_A = m * TILE_SIZE + threadIdx.x;
                if (load_col_A < N && row >= load_col_A)
                    sA[threadIdx.y][threadIdx.x] = A[row * N + load_col_A];
                else
                    sA[threadIdx.y][threadIdx.x] = 0.0f;

                // Load B tile
                int load_row_B = m * TILE_SIZE + threadIdx.y;
                if (load_row_B < N && load_row_B >= col)
                    sB[threadIdx.y][threadIdx.x] = B[load_row_B * N + col];
                else
                    sB[threadIdx.y][threadIdx.x] = 0.0f;

                __syncthreads();

                // Compute valid k range for this tile
                int k_start = max(col, m * TILE_SIZE);
                int k_end = min(row + 1, (m+1) * TILE_SIZE);
                int tiles_to_compute = k_end - k_start;

                #pragma unroll
                for (int k = 0; k < TILE_SIZE && k < tiles_to_compute; ++k) {
                    sum += sA[threadIdx.y][k + (k_start - m * TILE_SIZE)] * 
                           sB[k + (k_start - m * TILE_SIZE)][threadIdx.x];
                }
                __syncthreads();
            }
            C[row * N + col] = sum;
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // Configure kernel for maximum occupancy (adjust based on N)
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid;
    grid.x = max(1, min(32, (N + TILE_SIZE-1)/TILE_SIZE));
    grid.y = max(1, min(32, (N + TILE_SIZE-1)/TILE_SIZE));
    
    grid_strided_triangular_mm<<<grid, block>>>(A.data_ptr<float>(),
                                              B.data_ptr<float>(),
                                              C.data_ptr<float>(),
                                              N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Grid-strided triangular matrix multiplication");
}