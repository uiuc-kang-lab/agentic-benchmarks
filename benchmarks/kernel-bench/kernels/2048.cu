#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 32
#define VECTOR_SIZE 4

__device__ __forceinline__ void load_vectorized(const float* src, float4* dst, int row, int col, int N) {
    int idx = row * N + col;
    if (col + VECTOR_SIZE <= N && row < N)
        *dst = *reinterpret_cast<const float4*>(&src[idx]);
    else
        *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__global__ void strided_vectorized_triangular_mm(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float* __restrict__ C,
                                                const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE+1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int linear_idx = thread_id; linear_idx < N * N; linear_idx += stride) {
        const int row = linear_idx / N;
        const int col = linear_idx % N;
        
        if (row < col) {
            C[linear_idx] = 0.0f;
            continue;
        }

        float sum = 0.0f;
        for (int k_start = col; k_start <= row; k_start += TILE_SIZE) {
            const int k_end = min(k_start + TILE_SIZE, row + 1);
            
            // Load tile for k in [k_start, min(k_start + TILE_SIZE, row+1))
            const int tk = threadIdx.x % (TILE_SIZE/VECTOR_SIZE);
            const int ty = threadIdx.x / (TILE_SIZE/VECTOR_SIZE);
            
            // Vectorize both A and B loads
            float4 vecA, vecB;
            load_vectorized(A, &vecA, row, k_start + tk * VECTOR_SIZE, N);
            load_vectorized(B, &vecB, k_start + ty * VECTOR_SIZE, col, N);
            
            As[ty][tk * VECTOR_SIZE] = vecA.x;
            As[ty][tk * VECTOR_SIZE + 1] = vecA.y;
            As[ty][tk * VECTOR_SIZE + 2] = vecA.z;
            As[ty][tk * VECTOR_SIZE + 3] = vecA.w;
            
            Bs[ty][tk * VECTOR_SIZE] = vecB.x;
            Bs[ty][tk * VECTOR_SIZE + 1] = vecB.y;
            Bs[ty][tk * VECTOR_SIZE + 2] = vecB.z;
            Bs[ty][tk * VECTOR_SIZE + 3] = vecB.w;
            
            __syncthreads();

            // Compute partial sum for current tile
            for (int k = 0; k < (k_end - k_start); k++) {
                if ((k_start + k) >= k_start && (k_start + k) <= row) {
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }
            __syncthreads();
        }
        C[row * N + col] = sum;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // Configure kernel for maximum occupancy
    int threads = 256;
    int blocks = (N * N + threads - 1) / threads;
    blocks = min(blocks, 256);  // Cap blocks to reduce launch overhead

    strided_vectorized_triangular_mm<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided vectorized triangular matrix multiplication (CUDA)");
}