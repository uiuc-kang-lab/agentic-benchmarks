#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 4

__global__ void non_atomic_coalesced_access_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int start_row,
    int end_row
) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y + start_row;

    if (ty >= end_row || tx >= N) return;

    if (ty < tx) {
        C[ty * N + tx] = 0.0f;
    } else {
        float sum = 0.0f;
        for (int k = tx; k <= ty; ++k) {
            sum += A[ty * N + k] * B[k * N + tx];
        }
        C[ty * N + tx] = sum;
    }
}

at::Tensor forward_coalesced_optimized(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::zeros_like(A); // Initialize C to zeros

    const dim3 threadsPerBlock(16, 16); // Smaller block to improve coalesced memory access
    
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    const int chunk_size = (N + NUM_STREAMS - 1) / NUM_STREAMS;

    for (int i = 0; i < NUM_STREAMS; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, N);
        if (start >= end) continue;

        dim3 blocks(
            (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (end - start + threadsPerBlock.y - 1) / threadsPerBlock.y
        );

        non_atomic_coalesced_access_kernel<<<blocks, threadsPerBlock, 0, streams[i]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            start,
            end
        );
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_coalesced_optimized, "Coalesced access triangular matmul (CUDA)");
}