#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <algorithm>

// Kernel that computes a chunk of C = A.T * B. This kernel computes rows [i_offset, i_offset + chunk_M) of C.
// A: shape (K, global_M) stored as (K, global_M), where element A(k, i) is at A[k * global_M + i]
// B: shape (K, N) with element B(k, j) at B[k * N + j]
// C: shape (global_M, N) with row-major storage: element C(i, j) at C[i * N + j]
__global__ void matMulStreamKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int K,
                                   int chunk_M,  // number of rows in this chunk
                                   int global_M, // total number of columns in A (and rows in C)
                                   int N,
                                   int i_offset) {
    // Compute local row index within the chunk
    int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (local_i < chunk_M && j < N) {
        int global_i = local_i + i_offset; // Map to global row index
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A is accessed as A(k, global_i) = A[k * global_M + global_i]
            // B is accessed as B(k, j) = B[k * N + j]
            sum += A[k * global_M + global_i] * B[k * N + j];
        }
        C[global_i * N + j] = sum;
    }
}

// The forward function is exposed via PyBind11. It partitions the work across multiple CUDA streams.
// Inputs:
//   A: Tensor of shape (K, global_M) [CUDA, float32]
//   B: Tensor of shape (K, N) [CUDA, float32]
// Returns:
//   C: Tensor of shape (global_M, N) computed as A.T * B.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are CUDA tensors and of type float32.
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    // A: (K, global_M) and B: (K, N)
    int K = A.size(0);
    int global_M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (global_M, N).
    auto C = torch::zeros({global_M, N}, torch::device(A.device()).dtype(A.dtype()));

    // We'll partition the computation of C's rows across multiple CUDA streams.
    const int THREADS = 16;

    // Determine number of streams. For simplicity, we use 4 streams.
    int num_streams = 4;
    int chunk = (global_M + num_streams - 1) / num_streams; // number of rows per stream

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaError_t err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }

    // Launch kernels concurrently on different streams.
    for (int i = 0; i < num_streams; ++i) {
        int i_offset = i * chunk;
        int chunk_M = std::min(chunk, global_M - i_offset);
        if (chunk_M <= 0) break;

        dim3 blockDim(THREADS, THREADS);
        dim3 gridDim((chunk_M + THREADS - 1) / THREADS, (N + THREADS - 1) / THREADS);

        const float* A_ptr = A.data_ptr<float>();
        const float* B_ptr = B.data_ptr<float>();
        float* C_ptr = C.data_ptr<float>();

        matMulStreamKernel<<<gridDim, blockDim, 0, streams[i]>>>(A_ptr, B_ptr, C_ptr, K, chunk_M, global_M, N, i_offset);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }

    // Synchronize all streams and destroy them
    for (int i = 0; i < num_streams; ++i) {
        cudaError_t err = cudaStreamSynchronize(streams[i]);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using overlapped streams (CUDA)");
}
