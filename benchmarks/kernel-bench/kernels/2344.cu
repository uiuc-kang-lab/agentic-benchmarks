#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int BLOCK_SIZE_M = 128;  // Number of rows per block
const int BLOCK_SIZE_N = 128;  // Number of columns per block
const int THREAD_SIZE_X = 8;   // Work per thread in X dimension
const int THREAD_SIZE_Y = 8;   // Work per thread in Y dimension

__global__ void matmul_transposed_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       const int M, const int N, const int K) {
    // Block index
    const int block_m = blockIdx.y;
    const int block_n = blockIdx.x;

    // Thread index within block
    const int thread_x = threadIdx.x;
    const int thread_y = threadIdx.y;

    // Starting position for this thread
    const int m_start = block_m * BLOCK_SIZE_M + thread_y;
    const int n_start = block_n * BLOCK_SIZE_N + thread_x;

    // Stride between elements processed by this thread
    const int m_stride = blockDim.y;
    const int n_stride = blockDim.x;

    // Accumulator registers for each output element computed by this thread
    float acc[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.0f};

    // Loop over K dimension in steps
    for (int k = 0; k < K; k++) {
        // Each thread processes THREAD_SIZE_Y x THREAD_SIZE_X elements
        #pragma unroll
        for (int m_idx = 0; m_idx < THREAD_SIZE_Y; m_idx++) {
            const int m = m_start + m_idx * m_stride;
            if (m >= M) continue;

            #pragma unroll
            for (int n_idx = 0; n_idx < THREAD_SIZE_X; n_idx++) {
                const int n = n_start + n_idx * n_stride;
                if (n >= N) continue;

                acc[m_idx][n_idx] += A[m * K + k] * B[n * K + k];
            }
        }
    }

    // Write results to global memory
    #pragma unroll
    for (int m_idx = 0; m_idx < THREAD_SIZE_Y; m_idx++) {
        const int m = m_start + m_idx * m_stride;
        if (m >= M) continue;

        #pragma unroll
        for (int n_idx = 0; n_idx < THREAD_SIZE_X; n_idx++) {
            const int n = n_start + n_idx * n_stride;
            if (n >= N) continue;

            C[m * N + n] = acc[m_idx][n_idx];
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Calculate grid dimensions
    dim3 threadsPerBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 numBlocks((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
                   (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    matmul_transposed_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B (CUDA)");
}