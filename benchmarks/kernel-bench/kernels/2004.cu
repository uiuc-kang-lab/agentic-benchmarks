#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 4
#define BLOCK_X 32
#define BLOCK_Y 16

__global__ void combined_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int start_row,
    int end_row
) {
    int row_stride = blockDim.y * gridDim.y;
    int col_stride = blockDim.x * gridDim.x;

    int initial_row = start_row + blockIdx.y * blockDim.y + threadIdx.y;
    int initial_col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int row = initial_row; row < end_row; row += row_stride) {
        for (int col = initial_col; col < N; col += col_stride) {
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
    }
}

at::Tensor forward_combined(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::zeros_like(A);

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    const int chunk_size = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    dim3 block_dims(BLOCK_X, BLOCK_Y);

    for (int i = 0; i < NUM_STREAMS; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, N);
        if (start >= end) continue;

        int grid_x = (N + block_dims.x - 1) / block_dims.x;
        int grid_y = (end - start + block_dims.y - 1) / block_dims.y;
        dim3 grid_dims(grid_x, grid_y);

        combined_kernel<<<grid_dims, block_dims, 0, streams[i]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            start,
            end
        );
    }

    cudaDeviceSynchronize();
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_combined, "Combined stream-stride triangular matmul (CUDA)");
}