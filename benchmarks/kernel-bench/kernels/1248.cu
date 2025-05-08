#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// This kernel maps one dot product (i.e., one output element) to a single warp.
// Each warp's threads load chunks of the L dimension in a strided fashion, accumulate a partial sum,
// and then perform an efficient warp-level reduction via __shfl_down_sync.

__global__ void einsum_kernel_warp_reduction(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    // Total number of output elements: each corresponding to one dot product
    int total = BATCH * I * J * K;
    
    // Each warp computes one output element.
    // Compute a global warp ID
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;  // lane within the warp
    
    if (warpId >= total) return;
    
    // Map the warpId to output indices (b, i, j, k)
    int idx = warpId;
    int k_idx = idx % K;
    idx /= K;
    int j_idx = idx % J;
    idx /= J;
    int i_idx = idx % I;
    int b_idx = idx / I;
    
    // Calculate the base offset for the vector A[b,i,j,:]
    int offsetA = ((b_idx * I + i_idx) * J + j_idx) * L;
    
    float sum = 0.0f;
    // Each thread in the warp processes a subset of the L dimension
    for (int l = lane; l < L; l += 32) {
        float a_val = A[offsetA + l];
        float b_val = B[l * K + k_idx];  // B is of shape [L, K]
        sum += a_val * b_val;
    }
    
    // Warp-level reduction using __shfl_down_sync
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // The first thread in the warp writes the final sum
    if (lane == 0) {
        C[warpId] = sum;
    }
}

// The forward function validates inputs, sets up grid dimensions, and launches the kernel.
// Here, each warp computes one output element, and we launch enough threads to cover all warps.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch between A and B");

    int BATCH = A.size(0);
    int I = A.size(1);
    int J = A.size(2);
    int L = A.size(3);
    int K = B.size(1);

    auto C = torch::zeros({BATCH, I, J, K}, A.options());
    int total_output = BATCH * I * J * K;  // total number of dot products

    // Each warp (32 threads) computes one dot product, so total threads needed:
    int total_threads = total_output * 32;
    const int threads_per_block = 256;  // e.g., 256 threads per block (i.e., 8 warps per block)
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    einsum_kernel_warp_reduction<<<blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with warp-level reduction (CUDA)");
}
