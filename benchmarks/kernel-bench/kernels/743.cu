#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Define tile size and a threshold below which we use our custom kernel
#define TILE_SIZE 16
#define MATRIX_SIZE_THRESHOLD 512

// This kernel uses a 2D grid and 2D blocks to map the threads directly
// onto rows and columns of the output matrix. This ensures that each thread
// computes one element of the output matrix using shared memory tiling.

__global__ void IndexingTiledMatmulKernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int M, int K, int N) {
    // Allocate shared memory for a tile of A and B
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    // Compute the row and column index of the element this thread will compute
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    
    // Total number of tiles needed in the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over tiles of A and B
    for (int t = 0; t < numTiles; t++) {
        // Calculate global indices for loading tiles
        int tiledA_col = t * TILE_SIZE + threadIdx.x;
        int tiledB_row = t * TILE_SIZE + threadIdx.y;

        // Load element from A (if within bounds), else 0
        A_tile[threadIdx.y][threadIdx.x] = (row < M && tiledA_col < K) ? A[row * K + tiledA_col] : 0.0f;
        
        // Load element from B (if within bounds), else 0
        B_tile[threadIdx.y][threadIdx.x] = (tiledB_row < K && col < N) ? B[tiledB_row * N + col] : 0.0f;

        // Synchronize to make sure the tile is loaded
        __syncthreads();

        // Multiply the two tiles together
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }

        // Synchronize to make sure that computation is done before loading new tiles
        __syncthreads();
    }

    // Write the result back to global memory if within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Forward function to select between the custom kernel and cuBLAS based on the matrix sizes
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());
    
    // Use the custom indexed kernel if matrices are small enough
    if (M <= MATRIX_SIZE_THRESHOLD && N <= MATRIX_SIZE_THRESHOLD) {
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                     (M + TILE_SIZE - 1) / TILE_SIZE);

        IndexingTiledMatmulKernel<<<gridDim, blockDim>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, K, N
        );
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    } else {
        // Fall back to cuBLAS for larger matrices
        static cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
            cublasCreate(&handle);
        }
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                            N, M, K, &alpha,
                                            B.data_ptr<float>(), N,
                                            A.data_ptr<float>(), K,
                                            &beta, C.data_ptr<float>(), N);
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLAS sgemm failed");
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tiled matrix multiplication with optimal thread and block indexing (CUDA)");
}
