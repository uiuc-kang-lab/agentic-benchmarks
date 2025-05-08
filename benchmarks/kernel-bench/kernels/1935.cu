#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define warp size and number of warps per block
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8

// Kernel: Each warp computes one element of the lower triangular matrix C
__global__ void triangular_mm_warp_reduce_kernel(const float* __restrict__ A,
                                                   const float* __restrict__ B,
                                                   float* __restrict__ C,
                                                   int N,
                                                   int T) {
    // Identify warp within block and lane within warp
    int warp_in_block = threadIdx.y;  // each block has WARPS_PER_BLOCK warps (blockDim.y)
    int lane = threadIdx.x;           // lane index within the warp (0 to 31)
    
    // Compute the global warp index over all blocks
    int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (global_warp >= T) return;  // T = number of lower triangular elements = N*(N+1)/2
    
    // Convert the flattened triangular index to (i, j) coordinates
    int idx = global_warp;
    // Using the triangular number formula: i = floor((sqrt(8*idx+1)-1)/2)
    float d = sqrtf(8.0f * (float)idx + 1.0f);
    int i = (int)((d - 1.0f) / 2.0f);
    int triangular_begin = i * (i + 1) / 2;
    int j = idx - triangular_begin;  // j is the offset within the i-th row
    
    float partial_sum = 0.0f;
    // Each thread in the warp handles a portion of the k-summation
    // Valid k: from j to i (since A and B are lower triangular)
    for (int k = j + lane; k <= i; k += WARP_SIZE) {
        partial_sum += A[i * N + k] * B[k * N + j];
    }
    
    // Perform warp-level reduction using shuffle instructions
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }
    
    // Lane 0 writes the final reduced sum to C
    if (lane == 0) {
        C[i * N + j] = partial_sum;
    }
}

// Note: The output tensor C is expected to have zeros in the upper triangular region.
// We initialize C with zeros so that only the lower triangular part is overwritten.

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");
    
    int N = A.size(0);
    // Initialize output with zeros to ensure the upper triangular part remains zero
    auto C = torch::zeros({N, N}, A.options());
    
    // Total number of lower triangular elements
    int T = N * (N + 1) / 2;
    
    // Launch kernel: each warp computes one lower-triangular element of C
    // Block dimensions: each block has WARPS_PER_BLOCK warps, each warp consists of 32 threads
    dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
    int grid_x = (T + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(grid_x);
    
    triangular_mm_warp_reduce_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        T
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication with warp-level reduction (CUDA)");
}
