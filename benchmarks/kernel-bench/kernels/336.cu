#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define CHUNK_K 4

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

    const int batch = blockIdx.z;
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    const float* a_batch = A + batch * M * K;
    const float* b_batch = B + batch * K * N;

    float sum = 0.0f;

    for (int t_base = 0; t_base < K; t_base += TILE_SIZE) {
        const int t = t_base + threadIdx.x * CHUNK_K;
        
        // Vectorized load of A with padding handling
        if (row < M && t + CHUNK_K <= K) {
            const float4 a_vec = *reinterpret_cast<const float4*>(&a_batch[row * K + t]);
            As[threadIdx.y][threadIdx.x * CHUNK_K] = a_vec.x;
            As[threadIdx.y][threadIdx.x * CHUNK_K + 1] = a_vec.y;
            As[threadIdx.y][threadIdx.x * CHUNK_K + 2] = a_vec.z;
            As[threadIdx.y][threadIdx.x * CHUNK_K + 3] = a_vec.w;
        } else {
            #pragma unroll
            for (int k = 0; k < CHUNK_K; ++k) {
                As[threadIdx.y][threadIdx.x * CHUNK_K + k] = (t + k < K && row < M) ? a_batch[row * K + t + k] : 0.0f;
            }
        }

        // Transposed load of B with vectorization
        if (col < N && t + CHUNK_K <= K) {
            const float4 b_vec = *reinterpret_cast<const float4*>(&b_batch[t * N + col]);
            Bs[threadIdx.x * CHUNK_K][threadIdx.y] = b_vec.x;
            Bs[threadIdx.x * CHUNK_K + 1][threadIdx.y] = b_vec.y;
            Bs[threadIdx.x * CHUNK_K + 2][threadIdx.y] = b_vec.z;
            Bs[threadIdx.x * CHUNK_K + 3][threadIdx.y] = b_vec.w;
        } else {
            #pragma unroll
            for (int k = 0; k < CHUNK_K; ++k) {
                Bs[threadIdx.x * CHUNK_K + k][threadIdx.y] = (t + k < K && col < N) ? b_batch[(t + k) * N + col] : 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += __fmul_rn(As[threadIdx.y][k], Bs[k][threadIdx.x]);
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
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