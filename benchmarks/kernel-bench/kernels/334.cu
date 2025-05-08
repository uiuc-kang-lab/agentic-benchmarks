#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// Constant memory for frequently accessed dimensions
__constant__ int d_batch_size;
__constant__ int d_M;
__constant__ int d_K;
__constant__ int d_N;

__global__ void constant_mem_bmm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int batch = blockIdx.z;
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Use constant memory for offset calculations
    const float* a_batch = A + batch * d_M * d_K;
    const float* b_batch = B + batch * d_K * d_N;

    float sum = 0.0f;

    for (int t = 0; t < d_K; t += TILE_SIZE) {
        // Load data into shared memory using constant memory for bounds checking
        if (row < d_M && (t + threadIdx.x) < d_K) {
            As[threadIdx.y][threadIdx.x] = __ldg(&a_batch[row * d_K + t + threadIdx.x]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((t + threadIdx.y) < d_K && col < d_N) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&b_batch[(t + threadIdx.y) * d_N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum = __fmaf_rn(As[threadIdx.y][i], Bs[i][threadIdx.x], sum);
        }

        __syncthreads();
    }

    if (row < d_M && col < d_N) {
        C[batch * d_M * d_N + row * d_N + col] = sum;
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

    // Copy dimensions to constant memory
    cudaMemcpyToSymbol(d_batch_size, &batch_size, sizeof(int));
    cudaMemcpyToSymbol(d_M, &M, sizeof(int));
    cudaMemcpyToSymbol(d_K, &K, sizeof(int));
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        batch_size
    );

    constant_mem_bmm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>()
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Constant memory optimized batched matrix multiplication (CUDA)");
}