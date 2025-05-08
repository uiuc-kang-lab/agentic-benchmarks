#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define TILE_SIZE 16

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

// This kernel uses tiling and transposes the tile for matrix B to improve memory coalescing
__global__ void matmul_tiled_coalesced_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    // Register to accumulate result - helps reduce register pressure
    float C_value = 0.0f;

    // Pre-calculate number of tiles
    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over tiles
    #pragma unroll 4
    for (int m = 0; m < num_tiles; ++m) {
        // Calculate indices once
        const int a_idx = row * N + m * TILE_SIZE + tx;
        const int b_idx = (m * TILE_SIZE + ty) * N + col;

        // Load tile from matrix A (memory is row-major, so this is coalesced)
        As[ty][tx] = (row < N && m * TILE_SIZE + tx < N) ? A[a_idx] : 0.0f;

        // Load tile from matrix B and store transposed
        Bs[tx][ty] = (m * TILE_SIZE + ty < N && col < N) ? B[b_idx] : 0.0f;

        __syncthreads();

        // Compute partial product using the transposed B tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            C_value = __fmaf_rn(As[ty][k], Bs[tx][k], C_value);
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = C_value;
}

// Pybind interface
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

    matmul_tiled_coalesced_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_data, B_data, C_data, N);

    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication kernel (CUDA) with coalesced memory access for B");
}
