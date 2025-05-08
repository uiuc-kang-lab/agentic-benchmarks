#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int TILE_SIZE = 16;
__constant__ float B_const[1048576]; // 4MB constant memory limit on most GPUs

__global__ void matmul_transposed_kernel(const float* A, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int m = by * TILE_SIZE + ty;
    int n = bx * TILE_SIZE + tx;

    float c_val = 0.0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int k_offset = t * TILE_SIZE;

        // Load A tile
        if (m < M && (k_offset + tx) < K) {
            As[ty][tx] = A[m * K + k_offset + tx];
        } else {
            As[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            if ((k_offset + k) < K) {
                // Access B from constant memory
                c_val += As[ty][k] * B_const[n * K + k_offset + k];
            }
        }

        __syncthreads();
    }

    if (m < M && n < N) {
        C[m * N + n] = c_val;
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

    // Check if B fits in constant memory
    TORCH_CHECK(N * K * sizeof(float) <= 1048576, "Matrix B too large for constant memory");

    // Copy B to constant memory
    cudaMemcpyToSymbol(B_const, B.data_ptr<float>(), N * K * sizeof(float));

    auto C = torch::empty({M, N}, A.options());
    
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);
    
    matmul_transposed_kernel<<<grid, block>>>(
        A.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B (CUDA)");
}