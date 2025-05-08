#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel to compute C = tril(A * B) using shared memory and warp-level primitives
__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float sharedB[32][32]; // Shared memory for B
    float sum = 0.f;

    for (int tileIdx = 0; tileIdx * 32 < N; ++tileIdx) {
        // Load a tile of B into shared memory
        int localCol = tileIdx * 32 + threadIdx.y;
        int localRow = tileIdx * 32 + threadIdx.x;
        if (localCol < N && col < N)
            sharedB[threadIdx.y][threadIdx.x] = B[localRow * N + col];
        else
            sharedB[threadIdx.y][threadIdx.x] = 0.f;

        __syncthreads();

        if (row < N && col < N && row >= col) {
            // Compute using the shared memory
            for (int k = 0; k < 32 && (tileIdx*32 + k) < N; ++k) {
                sum += A[row * N + tileIdx * 32 + k] * sharedB[k][threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (row < N && col < N && row >= col) {
        C[row * N + col] = sum;
    } else if (row < N && col < N) {
        C[row * N + col] = 0.f;
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

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + 31) / 32, (N + 31) / 32);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication with shared memory and warp-level reduction (CUDA)");
}