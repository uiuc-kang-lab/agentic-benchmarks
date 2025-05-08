#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define FULL_MASK 0xffffffff

// Each warp computes one element of C using warp-level primitives for reduction
__global__ void warp_shuffle_matmul_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int M, int K, int N) {
    // Calculate warp ID within the grid
    int warpId = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;

    // Each warp processes one output element
    int index = warpId;
    if (index >= M * N) return;
    int row = index / N;
    int col = index % N;

    float sum = 0.0f;

    // Each lane computes a partial sum over the K dimension in strides of warp size
    for (int k = lane; k < K; k += 32) {
        float a_val = A[row * K + k];
        float b_val = B[k * N + col];
        sum += a_val * b_val;
    }
    
    // Warp-level reduction using shuffle down
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    // Lane 0 writes the result
    if (lane == 0) {
        C[row * N + col] = sum;
    }
}

// Host function
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Input tensors must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    // Fallback to cuBLAS for large matrices
    if (M >= 512 && N >= 512 && K >= 512) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    B.data_ptr<float>(), N,
                    A.data_ptr<float>(), K,
                    &beta, C.data_ptr<float>(), N);
        cublasDestroy(handle);
    } else {
        // Launch kernel where each warp computes one C element
        int total_elements = M * N;
        int threads_per_block = 128; // 4 warps per block
        int warps_per_block = threads_per_block / 32;
        int total_warps_needed = total_elements;
        int blocks = (total_warps_needed + warps_per_block - 1) / warps_per_block;

        warp_shuffle_matmul_kernel<<<blocks, threads_per_block>>>(A.data_ptr<float>(),
                                                                   B.data_ptr<float>(),
                                                                   C.data_ptr<float>(),
                                                                   M, K, N);
        cudaDeviceSynchronize();
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Warp shuffle matrix multiplication (CUDA)");
}
