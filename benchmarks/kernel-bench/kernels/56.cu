#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// Define block dimensions
#define BLOCK_SIZE 16
#define TILE_DIM (BLOCK_SIZE * 2)  // 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

// Branchless min function
__device__ inline int safe_min(int a, int b) {
    return a < b ? a : b;
}

// Kernel using 2x2 register tiling with branchless shared memory loads to minimize warp divergence
__global__ void matmul_regtile_no_div_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int N) {
    // Calculate block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // Thread indices in block (0 to BLOCK_SIZE-1)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Each block computes a TILE_DIM x TILE_DIM (32x32) output tile
    // Each thread computes a 2x2 sub-tile
    int row = by * TILE_DIM + ty * 2;  // starting row for this thread's sub-tile
    int col = bx * TILE_DIM + tx * 2;  // starting col for this thread's sub-tile

    // Registers for the 2x2 sub-tile
    float regC00 = 0.0f, regC01 = 0.0f, regC10 = 0.0f, regC11 = 0.0f;

    // Shared memory tiles for A and B
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int numTiles = (N + TILE_DIM - 1) / TILE_DIM;
    int totalElements = TILE_DIM * TILE_DIM; // total elements per tile = 32*32
    int linearIndex = ty * BLOCK_SIZE + tx;  // flattened thread index in block

    for (int t = 0; t < numTiles; t++) {
        // Load tile of A in a branchless manner using clamped indices and a mask
        for (int i = linearIndex; i < totalElements; i += BLOCK_SIZE * BLOCK_SIZE) {
            int r = i / TILE_DIM;
            int c = i % TILE_DIM;
            int global_r = by * TILE_DIM + r;
            int global_c = t * TILE_DIM + c;
            int safe_r = safe_min(global_r, N - 1);
            int safe_c = safe_min(global_c, N - 1);
            // Compute mask: 1.0 if within bounds, 0.0 otherwise
            float mask = (global_r < N ? 1.0f : 0.0f) * (global_c < N ? 1.0f : 0.0f);
            sA[r][c] = A[safe_r * N + safe_c] * mask;
        }
        // Load tile of B in a branchless manner
        for (int i = linearIndex; i < totalElements; i += BLOCK_SIZE * BLOCK_SIZE) {
            int r = i / TILE_DIM;
            int c = i % TILE_DIM;
            int global_r = t * TILE_DIM + r;
            int global_c = bx * TILE_DIM + c;
            int safe_r = safe_min(global_r, N - 1);
            int safe_c = safe_min(global_c, N - 1);
            float mask = (global_r < N ? 1.0f : 0.0f) * (global_c < N ? 1.0f : 0.0f);
            sB[r][c] = B[safe_r * N + safe_c] * mask;
        }
        __syncthreads();

        // Compute partial products with register tiling
        #pragma unroll
    for (int k = 0; k < TILE_DIM; k++) {
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

    // Write back the results with boundary checks (the divergence here is minimal as it only affects border threads)
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
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int64_t N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE); // 16x16 threads per block
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    matmul_regtile_no_div_kernel<<<blocks, threads>>>(A_data, B_data, C_data, N);
    C10_CUDA_CHECK(cudaGetLastError());
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication kernel (CUDA) with 2x2 register tiling and branchless loading");
}
