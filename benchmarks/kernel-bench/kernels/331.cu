#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__constant__ int const_M, const_K, const_N, const_batch_size;

__global__ void constant_mem_bmm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE+1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int batch = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;

    const float* a_batch = A + batch * const_M * const_K;
    const float* b_batch = B + batch * const_K * const_N;

    int numTiles = (const_K + TILE_SIZE - 1) / TILE_SIZE;
    
    #pragma unroll
    for (int t = 0; t < numTiles; t++) {
        int A_col = t * TILE_SIZE + threadIdx.x;
        int B_row = t * TILE_SIZE + threadIdx.y;

        if (row < const_M && A_col < const_K)
            As[threadIdx.y][threadIdx.x] = a_batch[row * const_K + A_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (B_row < const_K && col < const_N)
            Bs[threadIdx.y][threadIdx.x] = b_batch[B_row * const_N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if (row < const_M && col < const_N)
        C[batch * const_M * const_N + row * const_N + col] = sum;
}

torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    // Copy parameters to constant memory
    cudaMemcpyToSymbol(const_M, &M, sizeof(int));
    cudaMemcpyToSymbol(const_K, &K, sizeof(int));
    cudaMemcpyToSymbol(const_N, &N, sizeof(int));
    cudaMemcpyToSymbol(const_batch_size, &batch_size, sizeof(int));

    auto C = torch::zeros({batch_size, M, N}, A.options());

    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE,
                batch_size);
    constant_mem_bmm_kernel<<<blocks, dim3(TILE_SIZE, TILE_SIZE)>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>()
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched MM with constant memory");
}
