#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses warp-level primitives for reduction to compute C = A * B^T
// Each warp computes one output element C[i, j] by processing the dot product of row i of A and row j of B
// (recall that B is given in non-transposed form but acts as if transposed, so C[i, j] = dot(A[i,:], B[j,:])).

__global__ void warp_matmul_optimized_v2_kernel(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 int M, int N, int K) {
    // Thread index within the warp and warp index within the block
    unsigned int lane   = threadIdx.x; // lane id in [0,31]
    unsigned int warpId = threadIdx.y; // warp id within the block

    // Map each warp to one output element C[i, j]
    int i = blockIdx.y * blockDim.y + warpId; // row index
    int j = blockIdx.x;                      // column index

    if (i < M && j < N) {
        float sum = 0.0f;
        // Using only warp-level primitives, remove explicit shared memory usage
        for (int k = lane; k < K; k += 32) {
            sum += __ldg(&A[i * K + k]) * __ldg(&B[j * K + k]);
        }
        
        // Accumulate results across warp
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // Lane 0 writes the result
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
    const int warpSize = 32;
    const int warpsPerBlock = 8;
    dim3 block(warpSize, warpsPerBlock);
    dim3 grid(N, (M + warpsPerBlock - 1) / warpsPerBlock);

    // Launch the kernel
    warp_matmul_optimized_v2_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-level optimized matrix multiplication with transposed B (CUDA)");
}
