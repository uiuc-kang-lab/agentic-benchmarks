#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_SIZE 32
#define BLOCK_SIZE WARP_SIZE

template<typename T>
__device__ __forceinline__ T ldg(const T* ptr) {
    return __ldg(ptr);
}

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= N || col >= N) return;
    
    const int block_row_start = blockIdx.y * blockDim.y;
    const int block_col_start = blockIdx.x * blockDim.x;
    const int block_row_end = min(block_row_start + blockDim.y - 1, N-1);
    const int block_col_end = min(block_col_start + blockDim.x - 1, N-1);

    if (block_row_end < block_col_start) {
        C[row * N + col] = 0.f;
        return;
    }

    const int warp_row = row & ~(WARP_SIZE - 1);
    const int warp_col = col & ~(WARP_SIZE - 1);
    if (warp_row < warp_col) {
        C[row * N + col] = 0.f;
        return;
    }

    if (row < col) {
        C[row * N + col] = 0.f;
        return;
    }

    float sum = 0.f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        const int tile_start = t * TILE_SIZE;
        const int tile_end = min(tile_start + TILE_SIZE, N);
        
        if (tile_start > row) break;
        
        if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
            const int r = block_row_start + threadIdx.y;
            const int c = tile_start + threadIdx.x;
            As[threadIdx.y][threadIdx.x] = (r < N && c < N) ? ldg(&A[r * N + c]) : 0.0f;
            
            const int r2 = tile_start + threadIdx.y;
            const int c2 = block_col_start + threadIdx.x;
            Bs[threadIdx.y][threadIdx.x] = (r2 < N && c2 < N) ? ldg(&B[r2 * N + c2]) : 0.0f;
        }
        __syncthreads();
        
        if (row >= col) {
            const int k_start = max(tile_start, col);
            const int k_end = min(tile_end, row + 1);
            
            #pragma unroll 8
            for (int k = k_start; k < k_end; ++k) {
                const int tile_row = threadIdx.y;
                const int tile_col = threadIdx.x;
                const int k_local = k - tile_start;
                sum += As[tile_row][k_local] * Bs[k_local][tile_col];
            }
        }
        __syncthreads();
    }
    
    if (row >= col) {
        C[row * N + col] = sum;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Inputs must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Input dimensions must match");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    cudaMemcpyAsync(C.data_ptr<float>(), A.data_ptr<float>(), N * N * sizeof(float), cudaMemcpyDeviceToDevice, stream1);
    cudaMemcpyAsync(C.data_ptr<float>(), B.data_ptr<float>(), N * N * sizeof(float), cudaMemcpyDeviceToDevice, stream2);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    triangular_mm_kernel<<<blocks, threads, 0, stream3>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Triangular Matrix Multiplication with Streams (CUDA)");
}
