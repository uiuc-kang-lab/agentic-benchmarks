#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define constants: tile dimensions and vectorization width
#define TILE_SIZE 32
#define VECTOR_SIZE 4

// This kernel combines vectorized memory loads with an efficient lower-triangular tile mapping.
// We use a 1D grid over lower-triangular blocks computed via triangular number inversion and
// load data in vectorized float4 chunks to maximize memory throughput. Shared memory tiling
// (with extra column padding) is used to minimize bank conflicts.

__global__ void vt_triangular_mm(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int N) {
    // Map the linear blockIdx.x to lower-triangular tile indices (tile_i, tile_j).
    int blockId = blockIdx.x;
    float tmp = sqrtf(8.0f * (float)blockId + 1.0f);
    int tile_i = (int)((tmp - 1.0f) * 0.5f);
    int tile_j = blockId - tile_i * (tile_i + 1) / 2;

    int row_tile = tile_i * TILE_SIZE;
    int col_tile = tile_j * TILE_SIZE;

    // Block configuration: blockDim.x = TILE_SIZE/VECTOR_SIZE, blockDim.y = TILE_SIZE
    int local_row = threadIdx.y;          // [0, TILE_SIZE)
    int local_x = threadIdx.x;            // [0, TILE_SIZE/VECTOR_SIZE)

    // Global indices for the output tile. Each thread computes VECTOR_SIZE contiguous columns.
    int row = row_tile + local_row;
    int col_base = col_tile + local_x * VECTOR_SIZE;

    // Out of bounds check
    if (row >= N || col_base >= N)
        return;

    // We only compute lower-triangular values (row >= col). If thread computes an upper-triangular element, set it to 0.
    if (row < col_base) {
        #pragma unroll
        for (int v = 0; v < VECTOR_SIZE; v++) {
            if (col_base + v < N) {
                C[row * N + col_base + v] = 0.0f;
            }
        }
        return;
    }

    // Initialize accumulator using float4 for vectorized accumulation
    float4 sum;
    sum.x = sum.y = sum.z = sum.w = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Shared memory tiles for A and B. Adding an extra column to reduce bank conflicts.
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // Loop over tiles in the k dimension. Each tile covers a chunk of the inner summation index.
    for (int t = 0; t < numTiles; t++) {
        int tile_start = t * TILE_SIZE;

        // Only process tiles where k (the summation index) is within valid range (i.e. <= row).
        if (tile_start > row)
            break;

        // Load a tile from matrix A: we load row 'row' for columns [tile_start, tile_start+TILE_SIZE).
        int a_col = tile_start + local_x * VECTOR_SIZE;
        if (row < N) {
            if (a_col + VECTOR_SIZE <= N) {
                // Vectorized load using float4
                float4 a_vec = *reinterpret_cast<const float4*>(&A[row * N + a_col]);
                As[local_row][local_x * VECTOR_SIZE]     = a_vec.x;
                As[local_row][local_x * VECTOR_SIZE + 1] = a_vec.y;
                As[local_row][local_x * VECTOR_SIZE + 2] = a_vec.z;
                As[local_row][local_x * VECTOR_SIZE + 3] = a_vec.w;
            } else {
                // Handle boundary case
                for (int i = 0; i < VECTOR_SIZE; i++) {
                    int col_idx = a_col + i;
                    As[local_row][local_x * VECTOR_SIZE + i] = (col_idx < N) ? A[row * N + col_idx] : 0.0f;
                }
            }
        }

        // Load a tile from matrix B: we load row (tile_start + local_row) and columns [col_base, col_base+VECTOR_SIZE).
        int b_row = tile_start + local_row;
        if (b_row < N) {
            int b_col = col_base;
            if (b_col + VECTOR_SIZE <= N) {
                float4 b_vec = *reinterpret_cast<const float4*>(&B[b_row * N + b_col]);
                Bs[local_row][local_x * VECTOR_SIZE]     = b_vec.x;
                Bs[local_row][local_x * VECTOR_SIZE + 1] = b_vec.y;
                Bs[local_row][local_x * VECTOR_SIZE + 2] = b_vec.z;
                Bs[local_row][local_x * VECTOR_SIZE + 3] = b_vec.w;
            } else {
                for (int i = 0; i < VECTOR_SIZE; i++) {
                    int col_idx = b_col + i;
                    Bs[local_row][local_x * VECTOR_SIZE + i] = (col_idx < N) ? B[b_row * N + col_idx] : 0.0f;
                }
            }
        } else {
            // If the B row is out-of-bound, fill with zeros
            for (int i = 0; i < VECTOR_SIZE; i++) {
                Bs[local_row][local_x * VECTOR_SIZE + i] = 0.0f;
            }
        }

        __syncthreads();

        // Determine the k range for the dot product within the current tile.
        // We only need to accumulate for k in [max(tile_start, col_base), min(tile_start+TILE_SIZE, row+1)).
        int local_k_start = (tile_start < col_base) ? col_base - tile_start : 0;
        int local_k_end = (tile_start + TILE_SIZE <= row + 1) ? TILE_SIZE : (row - tile_start + 1);

        // Accumulate dot product: each iteration multiplies a value from A with a corresponding vector from B
        for (int k = local_k_start; k < local_k_end; k++) {
            float a_val = As[local_row][k];
            sum.x += a_val * Bs[k][local_x * VECTOR_SIZE];
            sum.y += a_val * Bs[k][local_x * VECTOR_SIZE + 1];
            sum.z += a_val * Bs[k][local_x * VECTOR_SIZE + 2];
            sum.w += a_val * Bs[k][local_x * VECTOR_SIZE + 3];
        }

        __syncthreads();
    }

    // Write the results back to matrix C for valid lower-triangular positions (row >= col).
    #pragma unroll
    for (int i = 0; i < VECTOR_SIZE; i++) {
        int col = col_base + i;
        if (col < N && row >= col) {
            C[row * N + col] = ((float*)&sum)[i];
        }
    }
}

// PyTorch forward interface
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same dimensions");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    // Total number of lower-triangular blocks
    int totalBlocks = numTiles * (numTiles + 1) / 2;
    // Launch configuration: each block is a tile of size TILE_SIZE x TILE_SIZE, vectorized along columns
    dim3 block(TILE_SIZE / VECTOR_SIZE, TILE_SIZE);
    dim3 grid(totalBlocks);

    vt_triangular_mm<<<grid, block>>>(A.data_ptr<float>(),
                                      B.data_ptr<float>(),
                                      C.data_ptr<float>(),
                                      N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized and tiled triangular matrix multiplication (CUDA)");
}
