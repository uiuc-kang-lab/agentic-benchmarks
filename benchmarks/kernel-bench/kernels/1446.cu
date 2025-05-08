#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use a larger tile size for better occupancy and memory reuse
#define TILE_SIZE 32
#define MAX_MATRIX_DIM 8192  // maximum supported matrix dimension

// Store frequently used constants in fast constant memory
__constant__ int d_N;
__constant__ int d_num_tiles;

// Optimized CUDA kernel using 2D thread blocks and tiling without costly atomics
__global__ void matmul_kernel_opt(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C) {
    // Map threads to a 2D tile in the output matrix
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float value = 0.0f;

    // Shared memory tiles for A and B
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    // Loop over tiles along the inner (k) dimension
    for (int t = 0; t < d_num_tiles; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        
        // Load the tile from matrix A
        if (row < d_N && a_col < d_N) {
            s_A[threadIdx.y][threadIdx.x] = A[row * d_N + a_col];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load the tile from matrix B
        if (b_row < d_N && col < d_N) {
            s_B[threadIdx.y][threadIdx.x] = B[b_row * d_N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles together; unroll the loop for performance
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            value += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result, with bounds checking
    if (row < d_N && col < d_N) {
        C[row * d_N + col] = value;
    }
}

// C++ interface (PyTorch binding)
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of same size");
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds maximum supported size");

    int N = A.size(0);
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Copy constants to fast constant memory
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));
    cudaMemcpyToSymbol(d_num_tiles, &num_tiles, sizeof(int));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Launch a 2D grid with 2D thread blocks matching the tile dimensions
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel_opt<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Matrix Multiplication (CUDA)");
}
