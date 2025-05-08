#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int TILE_SIZE = 8;
const int VECTOR_SIZE = 4;

__global__ void matmul_transposed_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Calculate base indices
    int row_start = by * TILE_SIZE + ty;
    int col_start = bx * TILE_SIZE + tx;

    // Each thread potentially handles multiple elements
    float c_vals[VECTOR_SIZE];
    #pragma unroll
    for (int v = 0; v < VECTOR_SIZE; v++) {
        c_vals[v] = 0.0f;
    }

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int k_offset = t * TILE_SIZE;

        // Collaborative loading of tiles
        #pragma unroll
        for (int v = 0; v < VECTOR_SIZE; v++) {
            int m = row_start + v * TILE_SIZE;
            if (m < M && (k_offset + tx) < K) {
                As[ty + v][tx] = A[m * K + k_offset + tx];
            } else {
                As[ty + v][tx] = 0.0f;
            }
        }

        #pragma unroll
        for (int v = 0; v < VECTOR_SIZE; v++) {
            int n = col_start + v * TILE_SIZE;
            if (n < N && (k_offset + ty) < K) {
                Bs[ty][tx + v] = B[n * K + k_offset + ty];
            } else {
                Bs[ty][tx + v] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int v = 0; v < VECTOR_SIZE; v++) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                c_vals[v] += As[ty][k] * Bs[k][tx + v];
            }
        }

        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int v = 0; v < VECTOR_SIZE; v++) {
        int m = row_start + v * TILE_SIZE;
        int n = col_start;
        if (m < M && n < N) {
            C[m * N + n] = c_vals[v];
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());
    
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);
    
    matmul_transposed_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B (CUDA)");
}