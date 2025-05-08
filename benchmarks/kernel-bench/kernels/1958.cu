#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define WARP_SIZE 32

// Each warp computes one lower-triangular output element using warp-level reduction
__global__ void warp_triangular_mm_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int N) {
    // Compute global thread id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;       // each warp handles one output element
    int lane = tid % WARP_SIZE;

    // Total number of lower-triangular elements
    int LT_count = N * (N + 1) / 2;
    if (warp_id >= LT_count) return;

    // Map warp_id to lower-triangular coordinates (i, j)
    // Using the formula: i = floor((sqrt(8*t + 1) - 1)/2) and j = t - i*(i+1)/2
    float tmp = sqrtf(8.0f * warp_id + 1.0f);
    int i = (int)((tmp - 1.0f) * 0.5f);
    int j = warp_id - (i * (i + 1)) / 2;

    // Compute dot product: C[i, j] = sum_{k = j}^{i} A[i, k] * B[k, j]
    int len = i - j + 1;  // number of terms in the summation
    float sum = 0.0f;
    for (int offset = lane; offset < len; offset += WARP_SIZE) {
        int k = j + offset;
        sum += A[i * N + k] * B[k * N + j];
    }

    // Reduce within the warp using __shfl_down_sync
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Lane 0 writes the result
    if (lane == 0) {
        C[i * N + j] = sum;
    }
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    // Allocate output tensor and initialize to zero so that the upper-triangular region is correct
    auto C = torch::zeros_like(A);

    // Total number of lower-triangular elements
    int LT_count = N * (N + 1) / 2;
    // Each warp (of 32 threads) computes one lower-triangular element.
    int total_threads = LT_count * WARP_SIZE;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    warp_triangular_mm_kernel<<<blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-Level Triangular Matrix Multiplication (CUDA)");
}
