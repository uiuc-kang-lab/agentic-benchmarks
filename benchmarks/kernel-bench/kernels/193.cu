#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define TILE_SIZE 16
#define SUB_TILE 2  // Each thread computes 2x2 output elements

__global__ void optimized_matrix_mult(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Each thread computes 2x2 output elements
    int row = by * TILE_SIZE + ty * SUB_TILE;
    int col = bx * TILE_SIZE + tx * SUB_TILE;

    float c[SUB_TILE][SUB_TILE] = {{0}};

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A (coalesced access)
        for (int i = 0; i < SUB_TILE; ++i) {
            int load_row = row + i;
            int load_col = t * TILE_SIZE + tx;
            if (load_row < M && load_col < K) {
                As[ty * SUB_TILE + i][tx] = A[load_row * K + load_col];
            } else {
                As[ty * SUB_TILE + i][tx] = 0.0f;
            }
        }

        // Load tile from B with transposed access pattern
        for (int i = 0; i < SUB_TILE; ++i) {
            int load_row = t * TILE_SIZE + ty;
            int load_col = col + i;
            if (load_row < K && load_col < N) {
                Bs[ty][tx * SUB_TILE + i] = B[load_row * N + load_col];
            } else {
                Bs[ty][tx * SUB_TILE + i] = 0.0f;
            }
        }

        __syncthreads();

        // Compute partial products
        for (int k = 0; k < (t + 1) * TILE_SIZE && k < K; ++k) {
            for (int i = 0; i < SUB_TILE; ++i) {
                float a = As[ty * SUB_TILE + i][k];
                for (int j = 0; j < SUB_TILE; ++j) {
                    c[i][j] += a * Bs[k][tx * SUB_TILE + j];
                }
            }
        }

        __syncthreads();
    }

    // Store results
    for (int i = 0; i < SUB_TILE; ++i) {
        for (int j = 0; j < SUB_TILE; ++j) {
            if ((row + i) < M && (col + j) < N) {
                C[(row + i) * N + (col + j)] = c[i][j];
            }
        }
    }
}

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    dim3 threads(TILE_SIZE / SUB_TILE, TILE_SIZE / SUB_TILE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    optimized_matrix_mult<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized tiled matrix multiplication (CUDA)");
}