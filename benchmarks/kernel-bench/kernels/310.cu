#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define BLOCK_SIZE 16

__device__ void load_tile_to_shared(
    const float* __restrict__ src,
    float* __restrict__ dest,
    int row, int col,
    int stride, int width,
    int height
) {
    if (row < height && col < width) {
        dest[threadIdx.y * TILE_SIZE + threadIdx.x] = src[row * stride + col];
    } else {
        dest[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
    }
}

__device__ float compute_tile_element(
    const float* __restrict__ tileA,
    const float* __restrict__ tileB
) {
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += tileA[threadIdx.y * TILE_SIZE + k] *
               tileB[k * TILE_SIZE + threadIdx.x];
    }
    return sum;
}

__global__ void bmm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    const int batch_idx = blockIdx.z;
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    const float* batch_A = A + batch_idx * M * K;
    const float* batch_B = B + batch_idx * K * N;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        load_tile_to_shared(
            batch_A,
            (float*)sharedA,
            row,
            tile * TILE_SIZE + threadIdx.x,
            K, TILE_SIZE,
            M
        );

        load_tile_to_shared(
            batch_B,
            (float*)sharedB,
            tile * TILE_SIZE + threadIdx.y,
            col,
            N, TILE_SIZE,
            K
        );

        __syncthreads();

        sum += compute_tile_element((float*)sharedA, (float*)sharedB);

        __syncthreads();
    }

    if (row < M && col < N) {
        C[batch_idx * M * N + row * N + col] = sum;
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

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        batch_size
    );

    bmm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication (CUDA)");
}