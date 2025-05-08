#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define VECTOR_WIDTH 4

typedef float float4 __attribute__((ext_vector_type(VECTOR_WIDTH)));

template <typename scalar_t>
__global__ void matmul_vectorized_kernel(const scalar_t* __restrict__ A,
                                       const scalar_t* __restrict__ B,
                                       scalar_t* __restrict__ C,
                                       int M, int K, int N) {
    __shared__ float4 sA[TILE_SIZE][TILE_SIZE/VECTOR_WIDTH];
    __shared__ float4 sB[TILE_SIZE][TILE_SIZE/VECTOR_WIDTH];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float4 accum[VECTOR_WIDTH] = {0};

    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        // Vectorized loading with aligned accesses
        if (row < M && (t * TILE_SIZE + threadIdx.x * VECTOR_WIDTH) < K) {
            const scalar_t* a_ptr = &A[row * K + t * TILE_SIZE + threadIdx.x * VECTOR_WIDTH];
            sA[threadIdx.y][threadIdx.x] = *reinterpret_cast<const float4*>(a_ptr);
        }

        if (col < N && (t * TILE_SIZE + threadIdx.x * VECTOR_WIDTH) < K) {
            const scalar_t* b_ptr = &B[(t * TILE_SIZE + threadIdx.x * VECTOR_WIDTH) * N + col];
            sB[threadIdx.x][threadIdx.y] = *reinterpret_cast<const float4*>(b_ptr);
        }

        __syncthreads();

        // Unrolled matrix multiply with uniform branching
        #pragma unroll
        for (int k = 0; k < TILE_SIZE/VECTOR_WIDTH; ++k) {
            float4 a = sA[threadIdx.y][k];
            float4 b = sB[k][threadIdx.x];
            
            #pragma unroll
            for (int vi = 0; vi < VECTOR_WIDTH; ++vi) {
                #pragma unroll
                for (int vj = 0; vj < VECTOR_WIDTH; ++vj) {
                    accum[vi] += a[vi] * b[vj];
                }
            }
        }
        __syncthreads();
    }

    // Coalesced write with vectorized stores
    if (row < M && col < N) {
        #pragma unroll
        for (int v = 0; v < VECTOR_WIDTH; ++v) {
            if (col + v < N) {
                C[row * N + col + v] = accum[v];
            }
        }
    }
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "Matrix dimensions mismatch");
    
    auto C = torch::zeros({M, N}, A.options());

    dim3 block(TILE_SIZE, TILE_SIZE/VECTOR_WIDTH);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, 
              (M + TILE_SIZE - 1) / TILE_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_vectorized", [&] {
        matmul_vectorized_kernel<scalar_t><<<grid, block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    });
    
    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Vectorized matrix multiplication (CUDA)");
}