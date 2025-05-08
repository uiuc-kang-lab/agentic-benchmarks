#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define VECTOR_SIZE 4

__global__ void optimized_triangular_mm(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE+1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col_base = blockIdx.x * TILE_SIZE + threadIdx.x * VECTOR_SIZE;

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (row < col_base && col_base < N) {
        for (int i = 0; i < VECTOR_SIZE; i++) {
            if (col_base + i < N) C[row * N + col_base + i] = 0.0f;
        }
        return;
    }

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int tile_start = t * TILE_SIZE;
        if (tile_start > row) break;

        // Coalesced load with vectorized access
        int a_col = tile_start + threadIdx.x * VECTOR_SIZE;
        if (a_col < N) {
            float4 vecA = __ldg(reinterpret_cast<const float4*>(&A[row * N + a_col]));
            As[threadIdx.y][threadIdx.x * VECTOR_SIZE] = vecA.x;
            As[threadIdx.y][threadIdx.x * VECTOR_SIZE + 1] = vecA.y;
            As[threadIdx.y][threadIdx.x * VECTOR_SIZE + 2] = vecA.z;
            As[threadIdx.y][threadIdx.x * VECTOR_SIZE + 3] = vecA.w;
        } else {
            for (int i = 0; i < VECTOR_SIZE; i++) {
                As[threadIdx.y][threadIdx.x * VECTOR_SIZE + i] = (a_col + i < N) ? A[row * N + a_col + i] : 0.0f;
            }
        }

        int b_row = tile_start + threadIdx.y;
        if (b_row < N) {
            float4 vecB = __ldg(reinterpret_cast<const float4*>(&B[b_row * N + col_base]));
            Bs[threadIdx.y][threadIdx.x * VECTOR_SIZE] = vecB.x;
            Bs[threadIdx.y][threadIdx.x * VECTOR_SIZE + 1] = vecB.y;
            Bs[threadIdx.y][threadIdx.x * VECTOR_SIZE + 2] = vecB.z;
            Bs[threadIdx.y][threadIdx.x * VECTOR_SIZE + 3] = vecB.w;
        } else {
            for (int i = 0; i < VECTOR_SIZE; i++) {
                Bs[threadIdx.y][threadIdx.x * VECTOR_SIZE + i] = (b_row < N && col_base + i < N) ? B[b_row * N + col_base + i] : 0.0f;
            }
        }

        __syncthreads();

        int k_start = max(tile_start, col_base);
        int k_end = min(tile_start + TILE_SIZE, row + 1);

        for (int k = k_start; k < k_end; ++k) {
            int k_tile = k - tile_start;
            float a_val = As[threadIdx.y][k_tile];
            sum.x += a_val * Bs[k_tile][threadIdx.x * VECTOR_SIZE];
            sum.y += a_val * Bs[k_tile][threadIdx.x * VECTOR_SIZE + 1];
            sum.z += a_val * Bs[k_tile][threadIdx.x * VECTOR_SIZE + 2];
            sum.w += a_val * Bs[k_tile][threadIdx.x * VECTOR_SIZE + 3];
        }

        __syncthreads();
    }

    if (row < N) {
        for (int i = 0; i < VECTOR_SIZE; i++) {
            int col = col_base + i;
            if (col < N && row >= col) {
                C[row * N + col] = ((float*)&sum)[i];
            }
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 block(TILE_SIZE / VECTOR_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    optimized_triangular_mm<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular matrix multiplication (CUDA)");
}
