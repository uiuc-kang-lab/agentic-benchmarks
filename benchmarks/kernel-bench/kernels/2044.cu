#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define tile and vector sizes
#define TILE_SIZE 32
#define VECTOR_SIZE 4

// Device helper: Vectorized load of a tile row from global memory into shared memory.
// Each thread loads VECTOR_SIZE contiguous elements if possible.
__device__ __forceinline__
void load_tile_vectorized(const float* __restrict__ src,
                            float dst[TILE_SIZE][TILE_SIZE+1],
                            int row, int col, int N, int stride) {
    float4 vec;
    int base_idx = row * stride + col;
    if (col + VECTOR_SIZE <= N && row < N) {
        vec = *reinterpret_cast<const float4*>(&src[base_idx]);
        dst[threadIdx.y][threadIdx.x * VECTOR_SIZE    ] = vec.x;
        dst[threadIdx.y][threadIdx.x * VECTOR_SIZE + 1] = vec.y;
        dst[threadIdx.y][threadIdx.x * VECTOR_SIZE + 2] = vec.z;
        dst[threadIdx.y][threadIdx.x * VECTOR_SIZE + 3] = vec.w;
    } else {
        #pragma unroll
        for (int i = 0; i < VECTOR_SIZE; i++) {
            dst[threadIdx.y][threadIdx.x * VECTOR_SIZE + i] = ((col + i < N) && (row < N)) ? src[base_idx + i] : 0.0f;
        }
    }
}

// Combined kernel that uses tiling with vectorized loads and a 1D grid mapping only of lower-triangular blocks.
// This kernel computes C = A*B for lower-triangular matrices (only computing C[i,j] for i >= j).
__global__ void tiled_vectorized_triangular_mm(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int N) {
    // Map the 1D block index into triangular tile coordinates using triangular number inversion.
    int blockId = blockIdx.x;
    float tmp = sqrtf(8.0f * (float)blockId + 1.0f);
    int tile_i = (int)((tmp - 1.0f) * 0.5f);
    int tile_j = blockId - (tile_i * (tile_i + 1) / 2);

    // Compute the global row and the base column for the 4 consecutive elements computed by each thread.
    int row = tile_i * TILE_SIZE + threadIdx.y;
    int col_base = tile_j * TILE_SIZE + threadIdx.x * VECTOR_SIZE;

    // If the row is outside the matrix, exit early.
    if (row >= N) return;

    // Each thread computes a vector of 4 partial sums
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Number of tiles along the k dimension
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Allocate shared memory for tiles of A and B. Extra column added to avoid bank conflicts.
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // Loop over tiles in the k dimension. For triangular matrices, the k-index runs from max(col, tile_start) to row+1.
    for (int t = 0; t < num_tiles; t++) {
        int tile_k_start = t * TILE_SIZE;
        // No contribution if the starting index of this tile exceeds the row index.
        if (tile_k_start > row) break;

        // Load a TILE_SIZE-wide tile from A that lies in row 'row' and columns from (tile_k_start) onward.
        load_tile_vectorized(A, As, row, tile_k_start + threadIdx.x * VECTOR_SIZE, N, N);
        // Load a tile from B: each thread loads VECTOR_SIZE elements from row (tile_k_start + threadIdx.y) and
        // column starting from the base computed from tile_j.
        load_tile_vectorized(B, Bs, tile_k_start + threadIdx.y, tile_j * TILE_SIZE + threadIdx.x * VECTOR_SIZE, N, N);

        __syncthreads();

        // Determine the effective range for k in this tile:
        // Start at the greater of the tile's beginning or the column base (lower bound of multiplication)
        int k_start = (tile_k_start > col_base) ? tile_k_start : col_base;
        // End at the minimum of the tile end and row+1 (since only indices <= row contribute for lower-triangular)
        int k_end = (tile_k_start + TILE_SIZE < (row + 1)) ? (tile_k_start + TILE_SIZE) : (row + 1);

        // Accumulate dot products for each element in the vector
        for (int k = k_start; k < k_end; k++) {
            int idx = k - tile_k_start;
            float a_val = As[threadIdx.y][idx];
            sum.x += a_val * Bs[idx][threadIdx.x * VECTOR_SIZE    ];
            sum.y += a_val * Bs[idx][threadIdx.x * VECTOR_SIZE + 1];
            sum.z += a_val * Bs[idx][threadIdx.x * VECTOR_SIZE + 2];
            sum.w += a_val * Bs[idx][threadIdx.x * VECTOR_SIZE + 3];
        }
        __syncthreads();
    }

    // Write out the computed results for each of the VECTOR_SIZE columns if they fall in the lower triangular region.
    #pragma unroll
    for (int i = 0; i < VECTOR_SIZE; i++) {
        int col = col_base + i;
        if (col < N && row >= col) {
            C[row * N + col] = ((float*)&sum)[i];
        }
    }
}

// Host interface: Performs error checking and kernel launch
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

    // Calculate number of tiles per matrix dimension and total lower-triangular blocks.
    int numTile = (N + TILE_SIZE - 1) / TILE_SIZE;
    int totalBlocks = numTile * (numTile + 1) / 2;

    // Launch configuration: using TILE_SIZE/Vector width horizontally and TILE_SIZE vertically
    dim3 block(TILE_SIZE / VECTOR_SIZE, TILE_SIZE);
    dim3 grid(totalBlocks);

    tiled_vectorized_triangular_mm<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined tiled vectorized triangular matrix multiplication (CUDA)");
}
