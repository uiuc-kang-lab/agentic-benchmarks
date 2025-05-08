#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses warp-level primitives for reduction to compute C = A * B^T
// Each warp computes one output element C[i, j] by processing the dot product of row i of A and row j of B
// (recall that B is given in non-transposed form but acts as if transposed, so C[i, j] = dot(A[i,:], B[j,:])).

__global__ void warp_matmul_memopt_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int M, int N, int K) {
    // Assumptions:
    //   - blockDim.x is the warp size (32)
    //   - blockDim.y is the number of warps per block
    // Each warp will compute one output element

    // Get lane index within the warp and warp index within the block
    unsigned int lane   = threadIdx.x; // lane id in [0,31]
    unsigned int warpId = threadIdx.y; // warp id within the block

    // Map each warp to one output element C[i, j]
    // Grid configuration: grid.x = N (one column per output element), grid.y covers rows by grouping warps
    int i = blockIdx.y * blockDim.y + warpId; // row index (from A)
    int j = blockIdx.x;                      // column index (corresponding to B's row, since C[i,j] = dot(A[i,:], B[j,:]))

    if (i < M && j < N) {
        float sum = 0.0f;
        // Using __ldg to optimize the read-only accesses to global memory
        for (int k = lane; k < K; k += 32) {
            float a = __ldg(&A[i * K + k]);
            float b = __ldg(&B[j * K + k]);
            sum += a * b;
        }
        
        // Perform warp-level reduction using __shfl_down_sync
        // Full warp mask
        unsigned int mask = 0xffffffff;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        
        // Lane 0 writes the result of the dot product.
        if (lane == 0) {
            C[i * N + j] = sum;
        }
    }
}

// Forward function callable from PyTorch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have the same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Configure launch parameters:
    // We'll use one warp (32 threads) per output element. Each block will have 32 threads in x and several warps in y.
    // For instance, choose 8 warps per block (i.e., blockDim.y = 8), so each block computes 8 output rows.
    const int warpSize = 32;
    const int warpsPerBlock = 8;
    dim3 block(warpSize, warpsPerBlock);
    // Grid: x-dimension covers output columns (N), y-dimension covers output rows in groups of (warpsPerBlock)
    dim3 grid(N, (M + warpsPerBlock - 1) / warpsPerBlock);

    // Launch the kernel
    warp_matmul_memopt_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-level accelerated matrix multiplication with transposed B using memory optimization (CUDA)");
}