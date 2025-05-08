#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>

#define TILE_SIZE 32
#define N_STREAMS 4

__global__ void upper_triangular_matmul_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N, int chunk_start) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + chunk_start;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        int start_k = row;
        int end_k = col;
        for (int k = start_k; k <= end_k; k++) {
            sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    std::vector<cudaStream_t> streams(N_STREAMS);
    for (int i = 0; i < N_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 threads(TILE_SIZE, TILE_SIZE);

    for (int chunk = 0; chunk < N; chunk += TILE_SIZE * N_STREAMS) {
        for (int i = 0; i < N_STREAMS; ++i) {
            int chunk_start = chunk + i * TILE_SIZE;
            if (chunk_start >= N) break;

            dim3 blocks((N + threads.x - 1) / threads.x, (TILE_SIZE + threads.y - 1) / threads.y);

            upper_triangular_matmul_kernel<<<blocks, threads, 0, streams[i]>>>(
                A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, chunk_start
            );
        }
    }

    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Optimized Upper Triangular Matrix Multiplication with Pipelined Streams");}