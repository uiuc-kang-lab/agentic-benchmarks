#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define UNROLL_FACTOR 4

__global__ void matmul_unrolled_warp_reduce_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x; 
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    const int TILE_M = 8;
    const int TILE_N = 4;

    const int warp_row = warp_id / TILE_N;
    const int warp_col = warp_id % TILE_N;

    const int m = blockIdx.y * TILE_M + warp_row;
    const int n = blockIdx.x * TILE_N + warp_col;

    float sum = 0.0f;

    // Manual unrolling of the K-dimension loop
    #pragma unroll 1
    for (int k_base = lane; k_base < K; k_base += WARP_SIZE * UNROLL_FACTOR) {
        float a_vals[UNROLL_FACTOR];
        float b_vals[UNROLL_FACTOR];
        
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; ++u) {
            const int k = k_base + u * WARP_SIZE;
            if (k < K) {
                if (m < M) {
                    a_vals[u] = __ldg(&A[m * K + k]);
                } else {
                    a_vals[u] = 0.0f;
                }
                if (n < N) {
                    b_vals[u] = __ldg(&B[n * K + k]);
                } else {
                    b_vals[u] = 0.0f;
                }
            } else {
                a_vals[u] = 0.0f;
                b_vals[u] = 0.0f;
            }
        }

        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; ++u) {
            sum += a_vals[u] * b_vals[u];
        }
    }

    // Warp-level reduction using __shfl_down_sync
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0 && m < M && n < N) {
        C[m * N + n] = sum;
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

    constexpr int TILE_M = 8;
    constexpr int TILE_N = 4;

    dim3 block(32, 32);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    matmul_unrolled_warp_reduce_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B using unrolled loops and warp-level reduction (CUDA)");
}