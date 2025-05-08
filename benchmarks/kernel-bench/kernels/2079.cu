#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Kernel that computes one element of the lower triangular output using warp-level reduction.
// Each warp (of 32 threads) is assigned one valid output element (i,j) in the lower triangular part,
// where the total number of valid elements is total = N*(N+1)/2. The (i,j) indices are computed
// from the warp's linear id using the inverse triangular number formula.
__global__ void triangular_mm_kernel_warp(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int N,
                                            int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if (warp_id >= total) return;

    // Compute (i, j) from warp_id using the inverse triangular number formula:
    // i = floor((sqrt(8*warp_id+1)-1)/2), j = warp_id - i*(i+1)/2.
    float tmp = sqrtf(8.0f * warp_id + 1.0f);
    int i = (int)((tmp - 1.0f) * 0.5f);
    int j = warp_id - i * (i + 1) / 2;
    if (i >= N || j >= N) return;

    float sum = 0.0f;
    // Each lane in the warp processes a subset of the reduction index range [j, i].
    for (int k = j + lane; k <= i; k += WARP_SIZE) {
        sum += A[i * N + k] * B[k * N + j];
    }
    // Perform warp-level reduction using __shfl_down_sync.
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    // Lane 0 writes the final result for element (i,j).
    if (lane == 0) {
        C[i * N + j] = sum;
    }
}

// Kernel to fill the upper triangular part of the output matrix with zero.
__global__ void fill_upper_kernel(float* __restrict__ C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.0f;
        }
    }
}

// Forward function exposed to PyTorch.
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Tensor A must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "Tensor B must be on CUDA");
    TORCH_CHECK(A.dim() == 2, "Tensor A must be 2D");
    TORCH_CHECK(B.dim() == 2, "Tensor B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "Tensor A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "Tensor B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Tensors A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);
    
    // Compute total number of valid elements in the lower triangular part (including diagonal).
    int total = N * (N + 1) / 2;
    
    // Launch the warp-level reduction kernel.
    // Each warp (of 32 threads) computes one output element; total threads = total * WARP_SIZE.
    int threads = 256;
    int blocks = (total * WARP_SIZE + threads - 1) / threads;
    triangular_mm_kernel_warp<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        total
    );
    
    // Launch a kernel to fill the upper triangular part with zero.
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + 31) / 32, (N + 31) / 32);
    fill_upper_kernel<<<numBlocks, threadsPerBlock>>>(C.data_ptr<float>(), N);
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-level reduction based lower triangular matrix multiplication (CUDA)");
}
