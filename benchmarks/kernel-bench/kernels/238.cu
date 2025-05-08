#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses warp-level primitives to reduce the dot-product
// without shared memory for the inner reduction.

// Each warp computes one output element of C. The warp distributes the work
// of computing the dot-product across its 32 lanes. Each lane processes a
// portion of the K dimension, and then a warp-level reduction is performed
// using __shfl_down_sync. 

__global__ void bmm_warp_shuffle_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    // Batch index
    int b = blockIdx.z;

    // Determine warp-level mapping:
    // Each block is configured with blockDim.x = 64 and blockDim.y = 4,
    // which gives 64/32 = 2 warps per row of the block and 4 rows => 8 warps per block.
    // Each warp computes one element of C.
    
    // warp_col: which warp column within the block (0 or 1)
    int warp_col = threadIdx.x / 32;  
    // lane index within the warp
    int lane = threadIdx.x % 32;
    // warp_row: which warp row within the block
    int warp_row = threadIdx.y;

    // Map warp to output element indices
    int m = blockIdx.y * blockDim.y + warp_row;  // row index in C
    int n = blockIdx.x * (blockDim.x / 32) + warp_col;  // column index in C

    float sum = 0.0f;

    // Only compute if m and n are within bounds
    if (m < M && n < N) {
        // Each lane processes elements in the K-dimension in stride of 32
        for (int k = lane; k < K; k += 32) {
            // A is of shape (batch_size, M, K)
            // B is of shape (batch_size, K, N)
            float a_val = A[b * M * K + m * K + k];
            float b_val = B[b * K * N + k * N + n];
            sum += a_val * b_val;
        }
        
        // Warp-level reduction using __shfl_down_sync to sum the partial results
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // The first lane stores the result
        if (lane == 0) {
            C[b * M * N + m * N + n] = sum;
        }
    }
}

// Host function to launch the kernel
// Mapping: Each warp computes one element of C.
// Block configuration: blockDim.x = 64, blockDim.y = 4.
// Each block computes 4 rows and 2 columns of output elements.

torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    // Block configuration: 
    //   blockDim.x = 64 (=> 2 warps per row), blockDim.y = 4 (4 warp rows per block)
    // Hence, each block computes 4 rows x 2 columns of output elements.
    dim3 block(64, 4);
    // Grid dimensions: 
    //   grid.x = ceil(N / 2), grid.y = ceil(M / 4), grid.z = batch_size
    dim3 grid((N + 2 - 1) / 2, (M + 4 - 1) / 4, batch_size);

    bmm_warp_shuffle_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication using warp-level shuffle (CUDA)");
}
