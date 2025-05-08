#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int TILE_SIZE = 32;

__global__ void matmul_transposed_kernel_minimal_sync(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float c_val = 0.0;
    
    // Pre-calculate indices
    const int tid = threadIdx.y * TILE_SIZE + threadIdx.x;
    const int total_threads = TILE_SIZE * TILE_SIZE;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int k_offset = t * TILE_SIZE;

        // Collaborative loading of tiles
        for (int i = tid; i < total_threads; i += total_threads) {
            int local_row = i / TILE_SIZE;
            int local_col = i % TILE_SIZE;
            
            // Load A tile
            int global_row = blockIdx.y * TILE_SIZE + local_row;
            if (global_row < M && (k_offset + local_col) < K) {
                As[local_row][local_col] = A[global_row * K + k_offset + local_col];
            } else {
                As[local_row][local_col] = 0.0;
            }

            // Load B tile
            int global_col = blockIdx.x * TILE_SIZE + local_col;
            if (global_col < N && (k_offset + local_row) < K) {
                Bs[local_row][local_col] = B[global_col * K + k_offset + local_row];
            } else {
                Bs[local_row][local_col] = 0.0;
            }
        }

        // Single sync after all loads
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            c_val += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Single sync before next iteration
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = c_val;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());
    
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);
    
    matmul_transposed_kernel_minimal_sync<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B minimal sync (CUDA)");
}