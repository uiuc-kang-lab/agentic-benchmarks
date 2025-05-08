#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 128
#define NUM_STREAMS 4

__global__ void triangular_mm_kernel_stride(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Use stride loops to handle large workloads
    for (int stride_row = row; stride_row < N; stride_row += blockDim.y * gridDim.y) {
        for (int stride_col = col; stride_col <= stride_row; stride_col += blockDim.x * gridDim.x) {
            if (stride_row < N && stride_col < N) {
                float sum = 0.f;
                for (int k = stride_col; k <= stride_row; ++k) {
                    sum += __ldg(&A[stride_row * N + k]) * __ldg(&B[k * N + stride_col]);
                }
                C[stride_row * N + stride_col] = sum;
            }
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 16;
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((N + threads - 1) / threads, (N + threads - 1) / threads);

    // Launch kernel with stride loops
    triangular_mm_kernel_stride<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}
