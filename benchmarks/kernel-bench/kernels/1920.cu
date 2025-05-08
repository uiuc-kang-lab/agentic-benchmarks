#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void strided_tril_mm_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by + ty;
    int col = bx + tx;
    
    float sum = 0.0f;
    
    for (int row_stride = row; row_stride < N; row_stride += blockDim.y * gridDim.y) {
        for (int col_stride = col; col_stride < N; col_stride += blockDim.x * gridDim.x) {
            if (row_stride >= col_stride) {
                for (int tile = col_stride / TILE_SIZE; tile <= row_stride / TILE_SIZE; ++tile) {
                    if ((tile * TILE_SIZE + tx) <= row_stride && row_stride < N) {
                        As[ty][tx] = A[row_stride * N + (tile * TILE_SIZE + tx)];
                    } else {
                        As[ty][tx] = 0.0f;
                    }
                    
                    if ((tile * TILE_SIZE + ty) <= N && col_stride < N) {
                        Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col_stride];
                    } else {
                        Bs[ty][tx] = 0.0f;
                    }
                    
                    __syncthreads();
                    
                    for (int k = 0; k < TILE_SIZE; ++k) {
                        if ((tile * TILE_SIZE + k) >= col_stride && (tile * TILE_SIZE + k) <= row_stride) {
                            sum += As[ty][k] * Bs[k][tx];
                        }
                    }
                    
                    __syncthreads();
                }

                if (row_stride < N && col_stride < N) {
                    C[row_stride * N + col_stride] = sum;
                }
            } else if (row_stride < N && col_stride < N) {
                C[row_stride * N + col_stride] = 0.0f;
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

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    strided_tril_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Strided triangular matrix multiplication (CUDA)");
}