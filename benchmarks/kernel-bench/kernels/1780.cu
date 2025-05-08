#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * blockDim.y + ty;
    const int col = blockIdx.x * blockDim.x + tx;

    float sum = 0.0f;

    if (row < N && col < N && row >= col) {
        for (int t = 0; t < (N + 15) / 16; ++t) {
            const int tile_idx = t * 16;
            if (tile_idx + tx <= row && tile_idx + ty < N) {
                As[ty][tx] = A[row * N + (tile_idx + tx)];
                Bs[ty][tx] = B[(tile_idx + ty) * N + col];
            } else {
                As[ty][tx] = 0.0f;
                Bs[ty][tx] = 0.0f;
            }
            
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                if (tile_idx + k >= col && tile_idx + k <= row) {
                    sum += As[ty][k] * Bs[k][tx];
                }
            }
            
            __syncthreads();
        }
        
        C[row * N + col] = sum;
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

    const int threads = 16;
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((N + threads - 1) / threads, (N + threads - 1) / threads);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular matrix multiplication (CUDA)");
}