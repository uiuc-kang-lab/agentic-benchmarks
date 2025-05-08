#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// Block size for threads per dimension
#define BLOCK_SIZE 16
// Each block computes a tile of size TILE_DIM x TILE_DIM, where TILE_DIM = 2 * BLOCK_SIZE
#define TILE_DIM (BLOCK_SIZE * 2)

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

// This kernel applies 2x2 register tiling. Each block computes a TILE_DIM x TILE_DIM chunk (here 32x32).
// The block is organized as BLOCK_SIZE x BLOCK_SIZE threads (16x16), and each thread computes a 2x2 submatrix.
// Shared memory is used to load tiles of A and B cooperatively via a grid-stride loop based on a flattened thread index.
__global__ void matmul_regtile_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    // Identify block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x; // Range: [0, BLOCK_SIZE)
    int ty = threadIdx.y; // Range: [0, BLOCK_SIZE)

    // Each block computes a TILE_DIM x TILE_DIM (=32x32) output tile.
    // Each thread computes a 2x2 sub-block.
    int row = by * TILE_DIM + ty * 2;  // starting row index for this thread's output
    int col = bx * TILE_DIM + tx * 2;  // starting col index for this thread's output

    // Registers to accumulate the 2x2 results
    float regC00 = 0.0f, regC01 = 0.0f, regC10 = 0.0f, regC11 = 0.0f;

    // Shared memory for the current tile of A and B
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    // Number of tiles along the k dimension
    int numTiles = (N + TILE_DIM - 1) / TILE_DIM;

    // Flatten thread index for cooperative loading into shared memory
    int linearIndex = ty * BLOCK_SIZE + tx; // each block has BLOCK_SIZE*BLOCK_SIZE threads
    int totalElements = TILE_DIM * TILE_DIM; // total elements to load per tile = 32*32 = 1024

    // Loop over tiles in the k dimension
    for (int t = 0; t < numTiles; t++) {
        // Load A tile into shared memory
        for (int i = linearIndex; i < totalElements; i += BLOCK_SIZE * BLOCK_SIZE) {
            int r = i / TILE_DIM;
            int c = i % TILE_DIM;
            int global_r = by * TILE_DIM + r;
            int global_c = t * TILE_DIM + c;
            sA[r][c] = (global_r < N && global_c < N) ? A[global_r * N + global_c] : 0.0f;
        }
        // Load B tile into shared memory
        for (int i = linearIndex; i < totalElements; i += BLOCK_SIZE * BLOCK_SIZE) {
            int r = i / TILE_DIM;
            int c = i % TILE_DIM;
            int global_r = t * TILE_DIM + r;
            int global_c = bx * TILE_DIM + c;
            sB[r][c] = (global_r < N && global_c < N) ? B[global_r * N + global_c] : 0.0f;
        }
        
        __syncthreads();

        // Compute the partial product for the 2x2 block computed by this thread
        // Each thread iterates over the k dimension of the tile
        for (int k = 0; k < TILE_DIM; k++) {
            // Each thread loads two elements from A and two elements from B
            float a0 = sA[ty * 2][k];
            float a1 = sA[ty * 2 + 1][k];
            float b0 = sB[k][tx * 2];
            float b1 = sB[k][tx * 2 + 1];
            regC00 += a0 * b0;
            regC01 += a0 * b1;
            regC10 += a1 * b0;
            regC11 += a1 * b1;
        }

        __syncthreads();
    }

    // Write the computed 2x2 block back to the global memory, checking boundaries
    if (row < N && col < N) {
        C[row * N + col] = regC00;
    }
    if (row < N && (col + 1) < N) {
        C[row * N + col + 1] = regC01;
    }
    if ((row + 1) < N && col < N) {
        C[(row + 1) * N + col] = regC10;
    }
    if ((row + 1) < N && (col + 1) < N) {
        C[(row + 1) * N + col + 1] = regC11;
    }
}

// Host function to launch the kernel
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);
    
    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int64_t N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE); // 16x16 threads per block
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    matmul_regtile_kernel<<<blocks, threads>>>(A_data, B_data, C_data, N);
    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication kernel (CUDA) with 2x2 register tiling");
}
