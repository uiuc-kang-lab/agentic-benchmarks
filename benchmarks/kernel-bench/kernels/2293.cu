#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// TILE_SIZE defines the dimensions of the tile (32x32)
#define TILE_SIZE 32

// This kernel computes C = A.T * B, where:
//   A is of shape (K, M) and is accessed in transposed fashion (i.e. A.T(i,k)=A(k,i)),
//   B is of shape (K, N), and
//   C is of shape (M, N).
// It uses the new asynchronous copy (cp.async) available on NVIDIA H100 to load global memory
// into shared memory, thereby overlapping memory loads with computation and reducing the number
// of explicit synchronizations (__syncthreads()).

// The kernel assumes the caller allocates enough shared memory for two tiles (one for A and one for B).

__global__ void matMulCpAsyncKernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int K, int M, int N) {
    // Compute global indices for the output matrix C
    int row = blockIdx.x * TILE_SIZE + threadIdx.y;  // corresponds to A's column index
    int col = blockIdx.y * TILE_SIZE + threadIdx.x;  // corresponds to B's column index

    float sum = 0.0f;

    // Allocate shared memory for one tile of A and one tile of B
    // We use externally allocated shared memory (via the kernel launch configuration)
    extern __shared__ float shared[];
    float* tileA = shared;                               // tile for A (size: TILE_SIZE*TILE_SIZE)
    float* tileB = shared + TILE_SIZE * TILE_SIZE;       // tile for B (size: TILE_SIZE*TILE_SIZE)

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aIndex = t * TILE_SIZE + threadIdx.x; // column index for A (global index in K dimension)
        int bIndex = t * TILE_SIZE + threadIdx.y; // row index for B (global index in K dimension)

        // Asynchronously copy one element from global memory to shared memory for A.
        // A is stored in row-major order as (K, M) and accessed in transposed manner: element = A[aIndex * M + row].
        if (row < M && aIndex < K) {
            asm volatile (
                "cp.async.cg.shared.global [%0], [%1], %2;\n"
                :
                : "r"(&tileA[threadIdx.y * TILE_SIZE + threadIdx.x]),
                  "l"(&A[aIndex * M + row]),
                  "n"(4)  // 4 bytes for a float
            );
        } else {
            tileA[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        }

        // Asynchronously copy one element from global memory to shared memory for B.
        // B is stored in row-major order as (K, N): element = B[bIndex * N + col].
        if (col < N && bIndex < K) {
            asm volatile (
                "cp.async.cg.shared.global [%0], [%1], %2;\n"
                :
                : "r"(&tileB[threadIdx.y * TILE_SIZE + threadIdx.x]),
                  "l"(&B[bIndex * N + col]),
                  "n"(4)
            );
        } else {
            tileB[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        }

        // Wait for the asynchronous copies to complete for this tile
        asm volatile("cp.async.wait_group 0;\n");
        // A minimal synchronization to make sure all threads see the loaded data
        __syncthreads();

        // Compute partial dot product for the current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + threadIdx.x];
        }

        // Synchronize before the next iteration overwrites the shared memory tiles
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// The forward function exposed via PyBind11
// Inputs:
//   A: Tensor of shape (K, M) [CUDA, float32]
//   B: Tensor of shape (K, N) [CUDA, float32]
// Returns:
//   C: Tensor of shape (M, N) computed as C = A.T * B

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (M, N)
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define thread block and grid sizes
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Allocate shared memory: we need space for 2 tiles (one for A and one for B)
    size_t sharedBytes = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the kernel
    matMulCpAsyncKernel<<<gridDim, blockDim, sharedBytes>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B (CUDA) using cp.async and minimal synchronizations");
}
