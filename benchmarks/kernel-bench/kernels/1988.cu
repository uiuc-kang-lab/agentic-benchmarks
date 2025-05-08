#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 4

__global__ void combined_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int start_row,
    int end_row
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y + start_row;

    if (row >= end_row || col >= N) return;

    if (row < col) {
        C[row * N + col] = 0.0f;
    } else {
        float sum = 0.0f;
        for (int k = col; k <= row; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

at::Tensor forward_combined(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::empty_like(A);

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Optimized block configuration from Kernel 1
    const int bx = 64, by = 16;
    dim3 threads(bx, by);
    const int chunk_size = (N + NUM_STREAMS - 1) / NUM_STREAMS;

    for (int i = 0; i < NUM_STREAMS; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, N);
        if (start >= end) continue;

        dim3 blocks(
            (N + bx - 1) / bx,
            (end - start + by - 1) / by
        );

        combined_kernel<<<blocks, threads, 0, streams[i]>>>(
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
    m.def("forward", &forward_combined, "Combined stream+optimized block triangular matmul (CUDA)");
}