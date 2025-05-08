#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void warp_optimized_mm_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles
    if (row >= col && row < N && col < N) {
        for (int tile = col / TILE_SIZE; tile <= row / TILE_SIZE; ++tile) {
            float As_val = (row < N && (tile * TILE_SIZE + threadIdx.x) <= row) ? A[row*N + tile*TILE_SIZE + threadIdx.x] : 0.0f;
            float Bs_val = (col < N && (tile * TILE_SIZE + threadIdx.y) < N) ? B[(tile * TILE_SIZE + threadIdx.y)*N + col] : 0.0f;
            
            for (int k = 0; k < TILE_SIZE; ++k) {
                if ((tile * TILE_SIZE + k) >= col && (tile * TILE_SIZE + k) <= row) {
                    sum += As_val * __shfl_sync(0xffffffff, Bs_val, k);
                }
            }
        }

        sum = warpReduceSum(sum);

        if (threadIdx.x % warpSize == 0) {
            C[row * N + col] = sum;
        }
    } else if (row < N && col < N) {
        C[row * N + col] = 0.0f;
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

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    warp_optimized_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Warp optimized triangular matrix multiplication (CUDA)");
}
