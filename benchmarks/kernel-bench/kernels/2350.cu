#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int TILE_SIZE = 16;

__global__ void matmul_transposed_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float c_val = 0.0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int k_offset = t * TILE_SIZE;

        if (row < M && (k_offset + tx) < K) {
            As[ty][tx] = A[row * K + k_offset + tx];
        } else {
            As[ty][tx] = 0.0;
        }

        if (col < N && (k_offset + ty) < K) {
            Bs[ty][tx] = B[col * K + k_offset + ty];
        } else {
            Bs[ty][tx] = 0.0;
        }

        __syncthreads();

        #pragma unroll
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            c_val = __fmaf_rn(As[ty][k], Bs[k][tx], c_val);
        }

        __syncthreads();
    }

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