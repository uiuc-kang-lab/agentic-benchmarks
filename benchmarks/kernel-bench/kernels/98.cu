#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define BLOCK_SIZE 32  // Tile size (each block computes a 32x32 tile of C)

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")


// Kernel that loads tiles from A and B using coalesced global memory accesses.
// Each block computes a 32x32 tile of C using a 16x16 thread block; each thread computes a 2x2 submatrix.
__global__ void balanced_coalesced_tiled_matmul_kernel(const float* __restrict__ A,
                                                       const float* __restrict__ B,
                                                       float* __restrict__ C,
                                                       int N) {
    // Shared memory for tiles of A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;  // 0..15
    int ty = threadIdx.y;  // 0..15
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Each thread computes a 2x2 submatrix of C
    int row = blockIdx.y * BLOCK_SIZE + ty * 2;  // starting row for this thread's submatrix
    int col = blockIdx.x * BLOCK_SIZE + tx * 2;    // starting col for this thread's submatrix

    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;

    // Loop over tiles in the K dimension
    for (int m = 0; m < N; m += BLOCK_SIZE) {
        // Load tile of A into shared memory with coalesced accesses
        // Each thread loads two elements from two rows of the A tile
        int aRow1 = blockIdx.y * BLOCK_SIZE + ty * 2;
        int aRow2 = aRow1 + 1;
        for (int i = tx; i < BLOCK_SIZE; i += blockDim.x) {
            int globalAcol = m + i;
            if (aRow1 < N && globalAcol < N)
                As[ty * 2][i] = A[aRow1 * N + globalAcol];
            else
                As[ty * 2][i] = 0.0f;
            if (aRow2 < N && globalAcol < N)
                As[ty * 2 + 1][i] = A[aRow2 * N + globalAcol];
            else
                As[ty * 2 + 1][i] = 0.0f;
        }

        // Load tile of B into shared memory with coalesced accesses
        // Each thread loads two elements from two rows of the B tile
        int bColBase = blockIdx.x * BLOCK_SIZE;
        for (int j = tx; j < BLOCK_SIZE; j += blockDim.x) {
            int globalBcol = bColBase + j;
            int bRow1 = m + ty * 2;
            int bRow2 = bRow1 + 1;
            if (bRow1 < N && globalBcol < N)
                Bs[ty * 2][j] = B[bRow1 * N + globalBcol];
            else
                Bs[ty * 2][j] = 0.0f;
            if (bRow2 < N && globalBcol < N)
                Bs[ty * 2 + 1][j] = B[bRow2 * N + globalBcol];
            else
                Bs[ty * 2 + 1][j] = 0.0f;
        }
        __syncthreads();

        // Compute the partial result for the 2x2 submatrix
        for (int k = 0; k < BLOCK_SIZE; k++) {
            float a0 = As[ty * 2][k];
            float a1 = As[ty * 2 + 1][k];
            float b0 = Bs[k][tx * 2];
            float b1 = Bs[k][tx * 2 + 1];
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        __syncthreads();
    }

    // Write the computed 2x2 submatrix to global memory with boundary checks
    if (row < N && col < N)
        C[row * N + col] = c00;
    if (row < N && (col + 1) < N)
        C[row * N + col + 1] = c01;
    if ((row + 1) < N && col < N)
        C[(row + 1) * N + col] = c10;
    if ((row + 1) < N && (col + 1) < N)
        C[(row + 1) * N + col + 1] = c11;
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
    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    // Launch configuration: each block computes a 32x32 tile using a 16x16 thread block
    dim3 threads(16, 16);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    balanced_coalesced_tiled_matmul_kernel<<<blocks, threads>>>(A_data, B_data, C_data, N);
    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced coalesced tiled matrix multiplication kernel (CUDA)");
}
