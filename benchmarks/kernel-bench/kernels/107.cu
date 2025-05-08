#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Each warp computes one output element of C = A * B.
// A is MxK, B is KxN, C is MxN and are stored in row-major order.
// Each thread in a warp processes a contiguous chunk of the inner dimension, then reduction is applied using __shfl_down_sync.

__global__ void matmul_warp_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Compute global thread id (using 1D grid)
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    // Each warp comprises 32 threads
    int warpId = tid / 32;
    int lane = tid % 32;

    // Map each warp to an output element in C
    // Total number of output elements is M * N
    int globalWarpId = warpId;  // Assuming launch configuration covers all elements
    int row = globalWarpId / N;
    int col = globalWarpId % N;

    if (row < M && col < N) {
        float sum = 0.0f;
        // Each lane in the warp processes a subset of the K dimension
        for (int k = lane; k < K; k += 32) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        // Warp-level reduction using __shfl_down_sync
        // All 32 lanes participate in the reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // Lane 0 writes the final result
        if (lane == 0) {
            C[row * N + col] = sum;
        }
    }
}


void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    // Ensure inputs are CUDA tensors and contiguous
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // Each warp computes one element of C.
    // Total elements in C:
    int totalElements = M * N;
    // Use 128 threads per block -> 128/32 = 4 warps per block
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / 32;
    // Total warps needed equals the total number of output elements
    int totalWarpsNeeded = totalElements;
    int numBlocks = (totalWarpsNeeded + warpsPerBlock - 1) / warpsPerBlock;

    // Launch the kernel
    matmul_warp_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in matmul_warp_kernel: %s\n", cudaGetErrorString(err));
    }
}

// The forward function allocates the output tensor and calls the custom matrix multiplication.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Allocate the output tensor on CUDA
    torch::Tensor C = torch::zeros({M, N}, A.options());

    matrix_multiply_cuda(A, B, C);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with optimized warp-level reduction");
}
