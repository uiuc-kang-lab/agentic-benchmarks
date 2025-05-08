#include <torch/extension.h>
#include <cuda_runtime.h>

// This kernel assigns one output element per warp. Each warp cooperatively computes the dot product
// along the L dimension using warp-level primitives (__shfl_down_sync) for reduction.
__global__ void einsum_kernel_warplevel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    // Total number of output elements
    int total = BATCH * I * J * K;
    
    // Each warp (32 threads) computes one output element
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;  // lane index within the warp
    
    if (warp_id >= total) return;

    // Decode the flat index into 4D coordinates for output C[b, i, j, k]
    int idx = warp_id;
    int k = idx % K;     
    idx /= K;
    int j = idx % J;
    idx /= J;
    int i = idx % I;
    int b = idx / I;
    
    // Each thread in the warp computes a partial sum of the dot product
    float sum = 0.0f;
    // Partition work over the L dimension, each lane handles a strided segment
    for (int l = lane; l < L; l += 32) {
        int a_index = b * (I * J * L) + i * (J * L) + j * L + l;
        int b_index = l * K + k;
        sum += A[a_index] * B[b_index];
    }
    
    // Perform warp-level reduction using __shfl_down_sync
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // The first lane writes the result
    if (lane == 0) {
        C[warp_id] = sum;
    }
}

// Forward function to launch the warp-level reduction kernel
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in L");

    int BATCH = A.size(0);
    int I = A.size(1);
    int J = A.size(2);
    int L = A.size(3);
    int K = B.size(1);

    // Create output tensor
    auto C = torch::zeros({BATCH, I, J, K}, A.options());

    // Total number of output elements to compute
    int total_out = BATCH * I * J * K;

    // Launch configuration: each warp (32 threads) computes one output element
    // Choose block size as 256 threads (8 warps per block)
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int blocks = (total_out + warps_per_block - 1) / warps_per_block;

    einsum_kernel_warplevel<<<blocks, threads_per_block>>>(
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
