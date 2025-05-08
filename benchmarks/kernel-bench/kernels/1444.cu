#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define MAX_MATRIX_DIM 8192  // Maximum supported matrix dimension

__constant__ int d_N;  // Matrix dimension in constant memory
__constant__ int d_num_tiles;  // Number of tiles needed for the computation

__global__ void matmul_kernel_optimized(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x % TILE_SIZE;
    const int ty = threadIdx.x / TILE_SIZE;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    float value = 0.0f;

    for (int m = 0; m < d_num_tiles; ++m) {
        const int a_idx = row * d_N + (m * TILE_SIZE + tx);
        const int b_idx = (m * TILE_SIZE + ty) * d_N + col;

        if (row < d_N && (m * TILE_SIZE + tx) < d_N)
            s_A[ty][tx] = A[a_idx];
        else
            s_A[ty][tx] = 0.0f;

        if ((m * TILE_SIZE + ty) < d_N && col < d_N)
            s_B[ty][tx] = B[b_idx];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            value += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    if (row < d_N && col < d_N) {
        C[row * d_N + col] = value;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of same size");
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds maximum supported size");

    const int N = A.size(0);
    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    cudaMemcpyToSymbol(d_N, &N, sizeof(int));
    cudaMemcpyToSymbol(d_num_tiles, &num_tiles, sizeof(int));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    const int threads_per_block = TILE_SIZE * TILE_SIZE;
    dim3 threads(threads_per_block);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaMemcpyAsync(A.data_ptr<float>(), A.data_ptr<float>(), N * N * sizeof(float), cudaMemcpyDeviceToDevice, stream1);
    cudaMemcpyAsync(B.data_ptr<float>(), B.data_ptr<float>(), N * N * sizeof(float), cudaMemcpyDeviceToDevice, stream2);

    matmul_kernel_optimized<<<blocks, threads, 0, stream3>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Matrix Multiplication with Streams (CUDA)");
}