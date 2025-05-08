#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define VECTOR_SIZE 4

typedef float float4_t __attribute__((ext_vector_type(VECTOR_SIZE)));

__global__ void vectorized_bmm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int batch = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;

    const float* a_batch = A + batch * M * K;
    const float* b_batch = B + batch * K * N;

    for (int t = 0; t < K; t += TILE_SIZE) {
        // Vectorized load for A
        if (row < M && t + threadIdx.x < K) {
            const float4_t* a_vec = reinterpret_cast<const float4_t*>(&a_batch[row * K + t + threadIdx.x * VECTOR_SIZE]);
            #pragma unroll
            for (int v = 0; v < VECTOR_SIZE; v++) {
                if (t + threadIdx.x * VECTOR_SIZE + v < K) {
                    As[threadIdx.y][threadIdx.x * VECTOR_SIZE + v] = a_vec[v];
                }
            }
        } else {
            #pragma unroll
            for (int v = 0; v < VECTOR_SIZE; v++) {
                As[threadIdx.y][threadIdx.x * VECTOR_SIZE + v] = 0.0f;
            }
        }

        // Vectorized load for B
        if (t + threadIdx.y < K && col < N) {
            const float4_t* b_vec = reinterpret_cast<const float4_t*>(&b_batch[(t + threadIdx.y) * N + col * VECTOR_SIZE]);
            #pragma unroll
            for (int v = 0; v < VECTOR_SIZE; v++) {
                if (col * VECTOR_SIZE + v < N) {
                    Bs[threadIdx.y][threadIdx.x * VECTOR_SIZE + v] = b_vec[v];
                }
            }
        } else {
            #pragma unroll
            for (int v = 0; v < VECTOR_SIZE; v++) {
                Bs[threadIdx.y][threadIdx.x * VECTOR_SIZE + v] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[batch * M * N + row * N + col] = sum;
    }
}

torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto C = torch::zeros({batch_size, M, N}, A.options());

    dim3 threads(TILE_SIZE / VECTOR_SIZE, TILE_SIZE);
    dim3 blocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        batch_size
    );

    vectorized_bmm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Vectorized batched matrix multiplication (CUDA)");
}