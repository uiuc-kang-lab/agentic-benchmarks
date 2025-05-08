#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define TILE_SIZE 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

// This kernel uses double buffering to prefetch the next tile into alternate shared memory buffers,
// allowing us to perform only one __syncthreads() per iteration for shared memory consistency.
__global__ void matmul_double_buffer_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N) {
    // Allocate two buffers for double buffering in shared memory
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    int currBuffer = 0;

    // Load the first tile (tile 0) into the current buffer
    int colA = 0 * TILE_SIZE + tx;
    int rowB = 0 * TILE_SIZE + ty;
    if (row < N && colA < N)
        As[currBuffer][ty][tx] = A[row * N + colA];
    else
        As[currBuffer][ty][tx] = 0.0f;

    if (rowB < N && col < N)
        Bs[currBuffer][ty][tx] = B[rowB * N + col];
    else
        Bs[currBuffer][ty][tx] = 0.0f;

    __syncthreads(); // Ensure the first tile is loaded

    float Cvalue = 0.0f;

    // Loop over tiles with double buffering
    for (int m = 0; m < numTiles - 1; m++) {
        int nextBuffer = 1 - currBuffer;
        int colA_next = (m + 1) * TILE_SIZE + tx;
        int rowB_next = (m + 1) * TILE_SIZE + ty;

        if (row < N && colA_next < N)
            As[nextBuffer][ty][tx] = A[row * N + colA_next];
        else
            As[nextBuffer][ty][tx] = 0.0f;

        if (rowB_next < N && col < N)
            Bs[nextBuffer][ty][tx] = B[rowB_next * N + col];
        else
            Bs[nextBuffer][ty][tx] = 0.0f;

        __syncthreads(); // Ensure the next tile is loaded before computing current tile

        // Compute multiplication for the current tile from the current buffer
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            Cvalue += As[currBuffer][ty][k] * Bs[currBuffer][k][tx];
        }
        
        // Switch to the prefetched tile
        currBuffer = nextBuffer;
        // No additional __syncthreads() here as previous barrier ensures all threads have loaded new tile
    }

    // Compute multiplication for the last tile
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
        Cvalue += As[currBuffer][ty][k] * Bs[currBuffer][k][tx];
    }

    if (row < N && col < N) {
        C[row * N + col] = Cvalue;
    }
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

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_double_buffer_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_data, B_data, C_data, N);

    C10_CUDA_CHECK(cudaGetLastError());
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication kernel (CUDA)");
}
