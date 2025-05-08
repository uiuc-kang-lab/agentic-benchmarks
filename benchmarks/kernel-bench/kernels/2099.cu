#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int N,
                                     int start_row,
                                     int end_row) {
    int row = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
    row += start_row;
    int col = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);

    if (row < end_row && col < N) {
        if (row < col) {
            C[row * N + col] = 0.0f;
        } else {
            float sum = 0.0f;
            #pragma unroll
            for (int k = col; k <= row; ++k) {
                sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
            }
            C[row * N + col] = sum;
        }
    }
}

at::Tensor forward(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be same size");

    const int N = static_cast<int>(A.size(0));
    auto C = torch::empty_like(A);

    const int threads = 16;
    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    const int chunk_size = (N + num_streams - 1) / num_streams;

    for (int i = 0; i < num_streams; ++i) {
        const int start_row = i * chunk_size;
        const int end_row = (i == num_streams-1) ? N : (i+1)*chunk_size;
        const int rows_in_chunk = end_row - start_row;

        dim3 blocks(
            (N + threads - 1) / threads,
            (rows_in_chunk + threads - 1) / threads
        );
        
        triangular_mm_kernel<<<blocks, dim3(threads, threads), 0, streams[i]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            start_row,
            end_row
        );
    }

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Multi-stream triangular matmul (CUDA)");
}