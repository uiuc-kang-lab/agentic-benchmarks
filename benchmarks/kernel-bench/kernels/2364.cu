#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile size for shared memory
#define TILE_SIZE 32

// Optimized kernel using double buffering to overlap memory loads with computation
__global__ void matmul_transposed_kernel_double_buffer(const float* __restrict__ A,
                                                         const float* __restrict__ B,
                                                         float* C,
                                                         int M, int N, int K) {
    // Allocate two buffers for double buffering
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute the global row and column indices of C
    int m = by * TILE_SIZE + ty;
    int n = bx * TILE_SIZE + tx;

    float c_val = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    int db_idx = 0;  // double buffering index

    // Preload the first tile into buffer 0
    int k_offset = 0;
    if(m < M && (k_offset + tx) < K)
        As[db_idx][ty][tx] = A[m * K + k_offset + tx];
    else
        As[db_idx][ty][tx] = 0.0f;

    if(n < N && (k_offset + tx) < K)
        Bs[db_idx][tx][ty] = B[n * K + k_offset + tx];
    else
        Bs[db_idx][ty][tx] = 0.0f;

    __syncthreads();

    // Loop over the tiles; use double buffering to load the next tile while computing the current tile
    for (int t = 0; t < numTiles - 1; t++) {
        int next_db = 1 - db_idx;
        int k_offset_next = (t + 1) * TILE_SIZE;

        // Load next A tile into the alternate shared memory buffer
        if(m < M && (k_offset_next + tx) < K)
            As[next_db][ty][tx] = A[m * K + k_offset_next + tx];
        else
            As[next_db][ty][tx] = 0.0f;

        // Load next B tile (B is stored row-major, but used as transposed) into its alternate buffer
        if(n < N && (k_offset_next + ty) < K)
            Bs[next_db][ty][tx] = B[n * K + k_offset_next + ty];
        else
            Bs[next_db][ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial product for the current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            c_val += As[db_idx][ty][k] * Bs[db_idx][k][tx];
        }

        // Switch buffers
        db_idx = next_db;
        __syncthreads();
    }

    // Final tile computation
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
        c_val += As[db_idx][ty][k] * Bs[db_idx][k][tx];
    }

    // Write back the result
    if(m < M && n < N) {
        C[m * N + n] = c_val;
    }
}


// Host function interfacing with PyTorch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have the same inner dimension (K)");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());
    
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);

    matmul_transposed_kernel_double_buffer<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized matrix multiplication with transposed B using double buffering (CUDA)");
}
