#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define TILE_K 64
#define WARP_SIZE 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")


// This kernel computes one element of the output matrix per warp. 
// Each warp cooperatively computes the dot product for one C element by partitioning the K dimension among its 32 lanes.
// A tile of B (a column vector segment) is loaded into shared memory to enable coalesced global memory loads.
// The partial products are reduced within the warp using __shfl_down_sync for an efficient final reduction.

__global__ void matmul_warp_tiled_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int N) {
    // Each warp in the block computes one output element C[row, col].
    const int lane = threadIdx.x;         // Lane index within the warp (0..31)
    const int warp_id = threadIdx.y;        // Warp index within the block
    const int row = blockIdx.y * blockDim.y + warp_id;
    const int col = blockIdx.x;             // Each block in x corresponds to one column of C

    float sum = 0.0f;

    // Allocate shared memory for a tile of B (a segment of the B column needed for current tile iteration)
    __shared__ float sB[TILE_K];

    // Number of tiles needed to cover the K dimension
    int numTiles = (N + TILE_K - 1) / TILE_K;

    for (int m = 0; m < numTiles; ++m) {
        int tile_base = m * TILE_K;
        // Load a tile of B (one column segment) cooperatively
        // Only threads in the x-dimension participate (stride of WARP_SIZE)
        // Pre-compute boundary conditions to reduce branch divergence
        const bool valid_col = col < N;
        for (int i = lane; i < TILE_K && (tile_base + i) < N; i += WARP_SIZE) {
            sB[i] = valid_col ? B[(tile_base + i) * N + col] : 0.0f;
        }
        __syncthreads(); // Ensure the B tile is loaded

        // Pre-compute row validity check
        const bool valid_row = row < N;
        float partial = 0.0f;
        if (valid_row) {
            // Each thread in the warp processes a subset of the tile elements in strides of WARP_SIZE
            const int tile_end = min(TILE_K, N - tile_base);
            for (int t = lane; t < tile_end; t += WARP_SIZE) {
                float a_val = A[row * N + (tile_base + t)];
                float b_val = sB[t];
                partial += a_val * b_val;
            }
            // Perform warp-level reduction using __shfl_down_sync to sum partial results
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                partial += __shfl_down_sync(0xffffffff, partial, offset);
            }
            // Only lane 0 of the warp adds the reduced partial sum to the overall sum
            if (lane == 0) {
                sum += partial;
            }
        }
        __syncthreads(); // Synchronize before loading the next B tile
    }

    // Write the final result. Only lane 0 of each warp writes out the computed C element.
    if (row < N && col < N && lane == 0) {
        C[row * N + col] = sum;
    }
}


// Pybind11 interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);

    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    // Configure the block and grid dimensions
    // Each block has 32 threads in x (one warp's lanes) and, e.g., 8 warps in y
    const int BLOCK_WARPS = 8;
    dim3 threads(WARP_SIZE, BLOCK_WARPS);
    // Each block computes one column for BLOCK_WARPS rows of C
    dim3 blocks(N, (N + BLOCK_WARPS - 1) / BLOCK_WARPS);

    matmul_warp_tiled_kernel<<<blocks, threads>>>(A_data, B_data, C_data, N);
    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication kernel (CUDA) with warp-level reduction and shared memory");
}
