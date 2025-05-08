#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void triangular_mm_kernel_optimized(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;

    if (row >= N || col >= N) return;
    
    // Handle upper triangle zeroing efficiently
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    // Calculate meaningful k range
    const int start_k = max(col, 0);
    const int end_k = min(row + 1, N);
    
    for (int kb = start_k; kb < end_k; kb += TILE_SIZE) {
        int k = kb + threadIdx.x;
        
        // Load A optimally with coalesced access
        if (k < end_k && row < N) {
            sA[threadIdx.y][threadIdx.x] = __ldg(&A[row * N + k]);
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B with main diagonal alignment
        int load_col = col;
        int k_b = kb + threadIdx.y;
        if (k_b < end_k && load_col < N) {
            sB[threadIdx.y][threadIdx.x] = __ldg(&B[k_b * N + load_col]);
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Accumulate valid range for this tile
        #pragma unroll
        for (int ki = 0; ki < TILE_SIZE; ++ki) {
            int global_k = kb + ki;
            if (global_k >= start_k && global_k < end_k) {
                sum += sA[threadIdx.y][ki] * sB[ki][threadIdx.x];
            }
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && A.size(0) == B.size(0), "Invalid dimensions");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE-1)/TILE_SIZE, (N + TILE_SIZE-1)/TILE_SIZE);

    triangular_mm_kernel_optimized<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular matmul");
}