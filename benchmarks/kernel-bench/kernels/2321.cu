#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define THREAD_WORK_M 4
#define THREAD_WORK_N 4

__global__ void matmul_balanced_workload_kernel(const float* A, const float* B, float* C, 
                                              const int M, const int N, const int K) {
    // Each thread computes multiple elements in both M and N dimensions
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    const int block_row = blockIdx.y * (BLOCK_SIZE * THREAD_WORK_M);
    const int block_col = blockIdx.x * (BLOCK_SIZE * THREAD_WORK_N);

    // Local accumulation registers
    float accum[THREAD_WORK_M][THREAD_WORK_N] = {0.0f};

    // Each thread processes multiple elements
    for (int k = 0; k < K; k++) {
        // Load values from A and B using read-only cache
        float a_vals[THREAD_WORK_M];
        float b_vals[THREAD_WORK_N];

        #pragma unroll
        for (int m = 0; m < THREAD_WORK_M; m++) {
            const int row = block_row + thread_row + m * BLOCK_SIZE;
            if (row < M) {
                a_vals[m] = __ldg(&A[row * K + k]);
            }
        }

        #pragma unroll
        for (int n = 0; n < THREAD_WORK_N; n++) {
            const int col = block_col + thread_col + n * BLOCK_SIZE;
            if (col < N) {
                b_vals[n] = __ldg(&B[col * K + k]);
            }
        }

        // Compute partial products
        #pragma unroll
        for (int m = 0; m < THREAD_WORK_M; m++) {
            #pragma unroll
            for (int n = 0; n < THREAD_WORK_N; n++) {
                accum[m][n] += a_vals[m] * b_vals[n];
            }
        }
    }

    // Write results to global memory
    #pragma unroll
    for (int m = 0; m < THREAD_WORK_M; m++) {
        const int row = block_row + thread_row + m * BLOCK_SIZE;
        if (row < M) {
            #pragma unroll
            for (int n = 0; n < THREAD_WORK_N; n++) {
                const int col = block_col + thread_col + n * BLOCK_SIZE;
                if (col < N) {
                    C[row * N + col] = accum[m][n];
                }
            }
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

    // Calculate grid dimensions based on thread work sizes
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (N + BLOCK_SIZE * THREAD_WORK_N - 1) / (BLOCK_SIZE * THREAD_WORK_N),
        (M + BLOCK_SIZE * THREAD_WORK_M - 1) / (BLOCK_SIZE * THREAD_WORK_M)
    );

    matmul_balanced_workload_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B using balanced workload (CUDA)");
}