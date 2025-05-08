#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Tile size for the kernel
#define TILE_SIZE 16

// This kernel multiplies matrices using tiled multiplication.
// It uses __ldg() for read-only loads of matrix A and B.
// For matrix B, it loads data in a vectorized way using float4 to ensure 128-bit aligned accesses.

__global__ void matrix_mul_ldg_aligned_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* C,
                                               int M, int N, int K) {
    // Shared memory for a tile of A (TILE_SIZE x TILE_SIZE)
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    // Shared memory for a tile of B stored as vectorized float4 (TILE_SIZE x (TILE_SIZE/4))
    __shared__ float4 sB[TILE_SIZE][TILE_SIZE/4];

    // Global row and column for C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load one element of A into shared memory using __ldg()
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            sA[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + a_col]);
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        /*
         For matrix B, we load a whole tile (of dimensions TILE_SIZE x TILE_SIZE) in a vectorized manner.
         We store the tile in shared memory as float4 arrays. Each float4 holds 4 consecutive floats,
         corresponding to 4 columns. There are TILE_SIZE/4 such groups per row.
         Only the first 64 threads (of the 256 threads in a 16x16 block) are used for loading B tile.
        */
        int linear_tid = threadIdx.y * blockDim.x + threadIdx.x; // 0 to 255
        int num_vec_elements = TILE_SIZE * (TILE_SIZE/4); // 16 * 4 = 64
        if (linear_tid < num_vec_elements) {
            int r = linear_tid / (TILE_SIZE/4); // r in [0, TILE_SIZE)
            int c = linear_tid % (TILE_SIZE/4);   // c in [0, TILE_SIZE/4)
            int b_row = t * TILE_SIZE + r;
            int b_col = blockIdx.x * TILE_SIZE + c * 4; // starting column for this vector load
            
            // Check boundaries for B. We want to load 4 consecutive floats at once.
            if (b_row < K && (b_col + 3) < N) {
                // Compute the index in float4 space. Assume that B is allocated with proper alignment.
                // Each row of B has N floats. In float4 units, the row length is N/4 (if N is divisible by 4).
                // If N is not a multiple of 4, we fallback below.
                if (N % 4 == 0) {
                    const float4* B_vec = reinterpret_cast<const float4*>(B);
                    int vec_index = b_row * (N / 4) + (b_col / 4);
                    sB[r][c] = __ldg(&B_vec[vec_index]);
                } else {
                    // Fallback: load element-wise if N is not divisible by 4
                    float tmp[4];
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int col_idx = b_col + i;
                        if (b_row < K && col_idx < N)
                            tmp[i] = __ldg(&B[b_row * N + col_idx]);
                        else
                            tmp[i] = 0.0f;
                    }
                    sB[r][c] = make_float4(tmp[0], tmp[1], tmp[2], tmp[3]);
                }
            } else {
                // Out-of-bound fallback: load scalars one by one
                float tmp[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int col_idx = b_col + i;
                    if (b_row < K && col_idx < N)
                        tmp[i] = __ldg(&B[b_row * N + col_idx]);
                    else
                        tmp[i] = 0.0f;
                }
                sB[r][c] = make_float4(tmp[0], tmp[1], tmp[2], tmp[3]);
            }
        }

        __syncthreads();

        // Compute the partial dot product for this tile.
        // Each thread computes one element C(row, col) using the loaded tiles.
        if (row < M && col < N) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                float a_val = sA[threadIdx.y][k];
                // For B, we need the element at row k and the appropriate column within the tile.
                // sB[k] is stored as an array of float4; each float4 holds 4 elements.
                int group = threadIdx.x / 4; // which float4 in the row
                int offset = threadIdx.x % 4; // which element within the float4
                float4 b_vec = sB[k][group];
                float b_val = (offset == 0) ? b_vec.x : ((offset == 1) ? b_vec.y : ((offset == 2) ? b_vec.z : b_vec.w));
                sum += a_val * b_val;
            }
        }

        __syncthreads();
    }

    // Write the result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


// Launcher: sets up the grid and block dimensions and launches the specialized kernel
void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    dim3 blockDim(TILE_SIZE, TILE_SIZE); // 16x16 threads per block
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matrix_mul_ldg_aligned_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
}

// Pybind interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Create output tensor without unnecessary initialization
    torch::Tensor C = torch::zeros({M, N}, A.options());
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with __ldg() and 128-bit aligned vectorized loads (CUDA)");
}
