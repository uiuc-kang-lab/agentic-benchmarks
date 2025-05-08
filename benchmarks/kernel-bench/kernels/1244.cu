#include <torch/extension.h>
#include <cuda_runtime.h>

// Define tile size for the K dimension
#define TILE_K 32

// This kernel computes C[b,i,j,k] = sum_{l} A[b,i,j,l] * B[l,k]
// Each block is mapped to a unique (b,i,j) element and a tile in the k dimension.
// Threads within a block process consecutive k values ensuring coalesced accesses for B and C.
__global__ void einsum_kernel_coalesced(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int IJP,  // Total number of (batch * I * J) groups
    int L,
    int K
) {
    // p indexes the (b, i, j) combination
    int p = blockIdx.x;
    // Each block processes a contiguous tile of the K dimension
    int k_base = blockIdx.y * TILE_K;
    int tid = threadIdx.x;
    int k = k_base + tid;

    // Only threads with valid k proceed
    if (k >= K) return;
    
    float sum = 0.0f;
    int offsetA = p * L;  // A is laid out as [BATCH*I*J, L]
    int offsetC = p * K;  // C is laid out as [BATCH*I*J, K]
    
    #pragma unroll
    for (int l = 0; l < L; ++l) {
        // Broadcast A element and coalesced access for B
        float a_val = A[offsetA + l];
        float b_val = B[l * K + k];
        sum += a_val * b_val;
    }
    C[offsetC + k] = sum;
}

// The forward function validates inputs, sets up grid dimensions, and launches the kernel
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
    
    auto C = torch::zeros({BATCH, I, J, K}, A.options());
    
    // Flatten the (BATCH, I, J) dimensions into one index
    int IJP = BATCH * I * J;
    
    // Set up a 2D grid: one dimension for (b,i,j) groups and one for tiled K
    dim3 blocks(IJP, (K + TILE_K - 1) / TILE_K);
    dim3 threads(TILE_K);

    einsum_kernel_coalesced<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        IJP, L, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with coalesced memory accesses (CUDA)");
}
