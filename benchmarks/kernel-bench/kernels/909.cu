#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel: Each warp cooperatively computes one output element of C = A * B
__global__ void warp_matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    const int warpSize = 32;
    // Compute unique warp id within the grid
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;
    int total_elements = M * N;

    // Each warp processes one (row, col) output element; use grid-stride loop over warps
    for (int idx = warp_id; idx < total_elements; idx += (gridDim.x * blockDim.x) / warpSize) {
         int row = idx / N;
         int col = idx % N;
         float sum = 0.0f;
         
         // Each lane in the warp takes care of part of the K dimension with a stride of warpSize
         for (int k = lane; k < K; k += warpSize) {
              float a = A[row * K + k];
              float b = B[k * N + col];
              sum += a * b;
         }
         
         // Perform warp-level reduction using __shfl_down_sync
         for (int offset = warpSize / 2; offset > 0; offset /= 2) {
              sum += __shfl_down_sync(0xffffffff, sum, offset);
         }
         
         // The first thread in the warp writes the final result
         if (lane == 0) {
              C[row * N + col] = sum;
         }
    }
}

// Host function interfacing with PyTorch
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");

    int M = A.size(0);
    int K = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Incompatible matrix dimensions");
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Configure block size and grid size: each block has a fixed number of threads (must be multiple of warpSize)
    int blockSize = 128; // e.g., 128 threads per block
    int warpsPerBlock = blockSize / 32; // number of warps in a block
    int total_elements = M * N;
    int gridSize = (total_elements + warpsPerBlock - 1) / warpsPerBlock;

    // Launch the kernel
    warp_matmul_kernel<<<gridSize, blockSize>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with warp-level reduction (CUDA)");
}
