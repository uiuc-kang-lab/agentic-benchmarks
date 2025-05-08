#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// TILE_SIZE is set to 16 for both block dimensions and shared memory tiles
#define TILE_SIZE 16

// Macro to check input tensor properties
#define CHECK_INPUT(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor"); \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous"); \
    TORCH_CHECK(x.scalar_type() == torch::kFloat, #x " must be a float tensor");

// CUDA kernel: uses vectorized loads with __ldg() and aligns accesses to 128-bit boundaries
__global__ void matrix_multiply_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int N, int K) {
    // Shared memory tiles for A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Calculate global row and column for C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float value = 0.0f;

    // Number of tiles in the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over tiles of A and B
    for (int t = 0; t < numTiles; t++) {
        int aTileColStart = t * TILE_SIZE;
        int bTileRow = t * TILE_SIZE + threadIdx.y;  // for B, the row in the tile
        
        // Number of 128-bit groups per row in the tile (each group is 4 floats)
        const int numGroups = TILE_SIZE / 4;  // For TILE_SIZE=16, numGroups = 4

        // Load tile of A into shared memory using vectorized loads
        // Each row of the tile from A: row index 'row', columns aTileColStart to aTileColStart+TILE_SIZE-1
        for (int grp = 0; grp < numGroups; grp++) {
            // Only one thread per group (the leader: those with threadIdx.x % 4 == 0 and matching group index)
            if ((threadIdx.x % 4) == 0 && (threadIdx.x / 4) == grp) {
                int colIndex = aTileColStart + grp * 4;
                float4 a_data;
                if (row < M) {
                    if (colIndex + 3 < K) {
                        // Aligned 128-bit load using __ldg()
                        a_data = __ldg(reinterpret_cast<const float4*>(&A[row * K + colIndex]));
                    } else {
                        // Handle boundary conditions with scalar loads
                        float tmp0 = (colIndex < K) ? __ldg(&A[row * K + colIndex]) : 0.0f;
                        float tmp1 = (colIndex + 1 < K) ? __ldg(&A[row * K + colIndex + 1]) : 0.0f;
                        float tmp2 = (colIndex + 2 < K) ? __ldg(&A[row * K + colIndex + 2]) : 0.0f;
                        float tmp3 = (colIndex + 3 < K) ? __ldg(&A[row * K + colIndex + 3]) : 0.0f;
                        a_data = make_float4(tmp0, tmp1, tmp2, tmp3);
                    }
                } else {
                    a_data = make_float4(0, 0, 0, 0);
                }
                // Write the loaded vector components into the shared memory tile
                int base = grp * 4;
                As[threadIdx.y][base + 0] = a_data.x;
                As[threadIdx.y][base + 1] = a_data.y;
                As[threadIdx.y][base + 2] = a_data.z;
                As[threadIdx.y][base + 3] = a_data.w;
            }
        }
        __syncthreads();

        // Load tile of B into shared memory using vectorized loads
        for (int grp = 0; grp < numGroups; grp++) {
            if ((threadIdx.x % 4) == 0 && (threadIdx.x / 4) == grp) {
                int colB = blockIdx.x * TILE_SIZE + grp * 4;
                float4 b_data;
                if ((t * TILE_SIZE + threadIdx.y) < K) { // row index for B in the tile
                    int b_row = t * TILE_SIZE + threadIdx.y;
                    if (colB + 3 < N) {
                        b_data = __ldg(reinterpret_cast<const float4*>(&B[b_row * N + colB]));
                    } else {
                        float tmp0 = (colB < N) ? __ldg(&B[b_row * N + colB]) : 0.0f;
                        float tmp1 = (colB + 1 < N) ? __ldg(&B[b_row * N + colB + 1]) : 0.0f;
                        float tmp2 = (colB + 2 < N) ? __ldg(&B[b_row * N + colB + 2]) : 0.0f;
                        float tmp3 = (colB + 3 < N) ? __ldg(&B[b_row * N + colB + 3]) : 0.0f;
                        b_data = make_float4(tmp0, tmp1, tmp2, tmp3);
                    }
                } else {
                    b_data = make_float4(0, 0, 0, 0);
                }
                int base = grp * 4;
                Bs[threadIdx.y][base + 0] = b_data.x;
                Bs[threadIdx.y][base + 1] = b_data.y;
                Bs[threadIdx.y][base + 2] = b_data.z;
                Bs[threadIdx.y][base + 3] = b_data.w;
            }
        }
        __syncthreads();

        // Compute partial dot product for the current tile
        for (int k = 0; k < TILE_SIZE; k++) {
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the computed value to global memory if within bounds
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Host function that wraps the kernel launch
void matrix_multiply_cuda(const torch::Tensor &A,
                            const torch::Tensor &B,
                            torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);

    matrix_multiply_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

// Pybind11 interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication (CUDA with vectorized __ldg loads and 128-bit alignment)");
}
