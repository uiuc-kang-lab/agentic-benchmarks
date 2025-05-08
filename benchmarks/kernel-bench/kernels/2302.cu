#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int TILE_SIZE = 32;
const int VECTOR_SIZE = 4;  // Use vector loads where possible

__global__ void matmul_transposed_kernel_coalesced(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float c_val = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int k_offset = t * TILE_SIZE;

        // Collaborative loading of A tile with coalesced access
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i += VECTOR_SIZE) {
            if (row < M && (k_offset + tx + i) < K && (tx + i) < TILE_SIZE) {
                float4 temp = *reinterpret_cast<const float4*>(&A[row * K + k_offset + tx + i]);
                As[ty][tx + i] = temp.x;
                if (i + 1 < TILE_SIZE) As[ty][tx + i + 1] = temp.y;
                if (i + 2 < TILE_SIZE) As[ty][tx + i + 2] = temp.z;
                if (i + 3 < TILE_SIZE) As[ty][tx + i + 3] = temp.w;
            } else {
                if ((tx + i) < TILE_SIZE) As[ty][tx + i] = 0.0f;
                if ((tx + i + 1) < TILE_SIZE) As[ty][tx + i + 1] = 0.0f;
                if ((tx + i + 2) < TILE_SIZE) As[ty][tx + i + 2] = 0.0f;
                if ((tx + i + 3) < TILE_SIZE) As[ty][tx + i + 3] = 0.0f;
            }
        }

        // Collaborative loading of B tile with coalesced access
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i += VECTOR_SIZE) {
            if (col < N && (k_offset + ty + i) < K && (ty + i) < TILE_SIZE) {
                float4 temp = *reinterpret_cast<const float4*>(&B[col * K + k_offset + ty + i]);
                Bs[ty + i][tx] = temp.x;
                if (i + 1 < TILE_SIZE) Bs[ty + i + 1][tx] = temp.y;
                if (i + 2 < TILE_SIZE) Bs[ty + i + 2][tx] = temp.z;
                if (i + 3 < TILE_SIZE) Bs[ty + i + 3][tx] = temp.w;
            } else {
                if ((ty + i) < TILE_SIZE) Bs[ty + i][tx] = 0.0f;
                if ((ty + i + 1) < TILE_SIZE) Bs[ty + i + 1][tx] = 0.0f;
                if ((ty + i + 2) < TILE_SIZE) Bs[ty + i + 2][tx] = 0.0f;
                if ((ty + i + 3) < TILE_SIZE) Bs[ty + i + 3][tx] = 0.0f;
            }
        }

        __syncthreads();

        // Compute dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            c_val += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write back result with coalesced access
    if (row < M && col < N) {
        C[row * N + col] = c_val;
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
    
    matmul_transposed_kernel_coalesced<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B coalesced (CUDA)");
}