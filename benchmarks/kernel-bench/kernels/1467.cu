#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 64
#define THREAD_TILE 4
#define MAX_MATRIX_DIM 8192

// Constant memory for matrix dimensions and number of tiles
__constant__ int d_N;
__constant__ int d_num_tiles;

// This kernel uses vectorized 128-bit aligned loads (float4) along with the __ldg() intrinsic
// to optimize read-only global memory access. Each thread computes a 4x4 sub-tile of the output matrix.

__global__ void vec_ldg_aligned_matmul(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C) {
    // Shared memory tiles for A and B
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    // The block computes a tile of size BLOCK_SIZE x BLOCK_SIZE in C
    // Each thread computes a 4x4 sub-tile. Thus, blockDim = (BLOCK_SIZE/THREAD_TILE, BLOCK_SIZE/THREAD_TILE) = (16, 16).
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute the starting global indices for the 4x4 output computed by this thread
    int rowStart = by * BLOCK_SIZE + ty * THREAD_TILE;
    int colStart = bx * BLOCK_SIZE + tx * THREAD_TILE;

    // Registers to accumulate a 4x4 sub-tile
    float regC[THREAD_TILE][THREAD_TILE] = { {0.f, 0.f, 0.f, 0.f},
                                               {0.f, 0.f, 0.f, 0.f},
                                               {0.f, 0.f, 0.f, 0.f},
                                               {0.f, 0.f, 0.f, 0.f} };

    // Loop over tiles in the k-dimension
    for (int t = 0; t < d_num_tiles; t++) {
        // Load tile from A into shared memory using vectorized loads.
        int total_A_loads = (BLOCK_SIZE * BLOCK_SIZE) / 4;
        int tid = ty * blockDim.x + tx;

        for (int i = tid; i < total_A_loads; i += (blockDim.x * blockDim.y)) {
            int a_row_in_tile = i / (BLOCK_SIZE / 4);
            int a_col_group = i % (BLOCK_SIZE / 4);

            int a_global_row = by * BLOCK_SIZE + a_row_in_tile;
            int a_global_col = t * BLOCK_SIZE + a_col_group * 4;

            float4 A_vec;
            if (a_global_row < d_N && (a_global_col + 3) < d_N) {
                const float4* A_vec_ptr = reinterpret_cast<const float4*>(A);
                int index = a_global_row * d_N + a_global_col;
                A_vec = __ldg(&A_vec_ptr[index / 4]);
            } else {
                float tmp[4] = {0.f, 0.f, 0.f, 0.f};
                for (int j = 0; j < 4; j++) {
                    int col = a_global_col + j;
                    if (a_global_row < d_N && col < d_N)
                        tmp[j] = __ldg(&A[a_global_row * d_N + col]);
                }
                A_vec.x = tmp[0];
                A_vec.y = tmp[1];
                A_vec.z = tmp[2];
                A_vec.w = tmp[3];
            }
            
            int dest_col = a_col_group * 4;
            s_A[a_row_in_tile][dest_col + 0] = A_vec.x;
            s_A[a_row_in_tile][dest_col + 1] = A_vec.y;
            s_A[a_row_in_tile][dest_col + 2] = A_vec.z;
            s_A[a_row_in_tile][dest_col + 3] = A_vec.w;
        }

        // Load tile from B into shared memory using vectorized loads
        int total_B_loads = (BLOCK_SIZE * BLOCK_SIZE) / 4;
        for (int i = tid; i < total_B_loads; i += (blockDim.x * blockDim.y)) {
            int b_row_in_tile = i / (BLOCK_SIZE / 4);
            int b_col_group = i % (BLOCK_SIZE / 4);
            int b_global_row = t * BLOCK_SIZE + b_row_in_tile;
            int b_global_col = bx * BLOCK_SIZE + b_col_group * 4;
            
            float4 B_vec;
            if (b_global_row < d_N && (b_global_col + 3) < d_N) {
                const float4* B_vec_ptr = reinterpret_cast<const float4*>(B);
                int index = b_global_row * d_N + b_global_col;
                B_vec = __ldg(&B_vec_ptr[index / 4]);
            } else {
                float tmp[4] = {0.f, 0.f, 0.f, 0.f};
                for (int j = 0; j < 4; j++) {
                    int col = b_global_col + j;
                    if (b_global_row < d_N && col < d_N)
                        tmp[j] = __ldg(&B[b_global_row * d_N + col]);
                }
                B_vec.x = tmp[0];
                B_vec.y = tmp[1];
                B_vec.z = tmp[2];
                B_vec.w = tmp[3];
            }
            int dest_col = b_col_group * 4;
            s_B[b_row_in_tile][dest_col + 0] = B_vec.x;
            s_B[b_row_in_tile][dest_col + 1] = B_vec.y;
            s_B[b_row_in_tile][dest_col + 2] = B_vec.z;
            s_B[b_row_in_tile][dest_col + 3] = B_vec.w;
        }

        __syncthreads();
        
        // Multiply the loaded tiles; each thread computes its 4x4 sub-block.
        int a_sub_row = ty * THREAD_TILE; // starting row in s_A for this thread
        int b_sub_col = tx * THREAD_TILE;   // starting col in s_B for this thread

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            float a0 = s_A[a_sub_row + 0][k];
            float a1 = s_A[a_sub_row + 1][k];
            float a2 = s_A[a_sub_row + 2][k];
            float a3 = s_A[a_sub_row + 3][k];

            float b0 = s_B[k][b_sub_col + 0];
            float b1 = s_B[k][b_sub_col + 1];
            float b2 = s_B[k][b_sub_col + 2];
            float b3 = s_B[k][b_sub_col + 3];

            regC[0][0] += a0 * b0;
            regC[0][1] += a0 * b1;
            regC[0][2] += a0 * b2;
            regC[0][3] += a0 * b3;

            regC[1][0] += a1 * b0;
            regC[1][1] += a1 * b1;
            regC[1][2] += a1 * b2;
            regC[1][3] += a1 * b3;

            regC[2][0] += a2 * b0;
            regC[2][1] += a2 * b1;
            regC[2][2] += a2 * b2;
            regC[2][3] += a2 * b3;

            regC[3][0] += a3 * b0;
            regC[3][1] += a3 * b1;
            regC[3][2] += a3 * b2;
            regC[3][3] += a3 * b3;
        }

        __syncthreads();
    }

    // Write the 4x4 sub-tile from registers back to global memory C using vectorized stores if possible
    for (int i = 0; i < THREAD_TILE; i++) {
        int global_row = rowStart + i;
        if (global_row < d_N) {
            int global_col = colStart;
            if (global_col + 3 < d_N) {
                float4 out_val;
                out_val.x = regC[i][0];
                out_val.y = regC[i][1];
                out_val.z = regC[i][2];
                out_val.w = regC[i][3];
                float4* C_vec_ptr = reinterpret_cast<float4*>(C);
                int index = global_row * d_N + global_col;
                C_vec_ptr[index / 4] = out_val;
            } else {
                for (int j = 0; j < THREAD_TILE; j++) {
                    int global_col_j = global_col + j;
                    if (global_col_j < d_N)
                        C[global_row * d_N + global_col_j] = regC[i][j];
                }
            }
        }
    }
}

// C++ interface using Pybind11

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same dimensions");
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds maximum supported size");

    int N = A.size(0);
    int num_tiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMemcpyToSymbol(d_N, &N, sizeof(int));
    cudaMemcpyToSymbol(d_num_tiles, &num_tiles, sizeof(int));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Launch configuration: blockDim = (BLOCK_SIZE/THREAD_TILE, BLOCK_SIZE/THREAD_TILE) = (16, 16)
    dim3 threads(BLOCK_SIZE / THREAD_TILE, BLOCK_SIZE / THREAD_TILE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    vec_ldg_aligned_matmul<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized 128-bit Aligned Matrix Multiplication with __ldg (CUDA)");
}
