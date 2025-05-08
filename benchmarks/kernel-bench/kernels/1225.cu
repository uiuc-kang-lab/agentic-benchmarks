#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel that computes C = einsum('bijl,lk->bijk') using warp-level reduction
// Each block computes one output element C[b,i,j,k] by reducing the dot product over the L dimension.

__global__ void einsum_kernel_warp_reduce(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    // Map blockIdx.x to output indices b, i, j, k
    int idx = blockIdx.x;
    int k = idx % K;
    int tmp = idx / K;
    int j = tmp % J;
    tmp /= J;
    int i = tmp % I;
    int b = tmp / I;

    int tid = threadIdx.x;
    float sum = 0.0f;

    // Each thread processes a slice of the L dimension.
    for (int l = tid; l < L; l += blockDim.x) {
        int a_index = b * (I * J * L) + i * (J * L) + j * L + l;
        int b_index = l * K + k;  // B is stored in row-major order
        sum += A[a_index] * B[b_index];
    }

    // Intra-warp reduction using warp shuffle primitives
    unsigned int full_mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(full_mask, sum, offset);
    }

    // Use shared memory to accumulate partial sums from each warp
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    __shared__ float sdata[32];  // Assuming blockDim.x <= 1024, so at most 32 warps per block

    if (lane == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction from each warp's partial sum by the first warp
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (tid < numWarps) {
        sum = sdata[lane];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(full_mask, sum, offset);
        }
        if (lane == 0) {
            int c_index = b * (I * J * K) + i * (J * K) + j * K + k;
            C[c_index] = sum;
        }
    }
}

// Forward function to launch the optimized kernel

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "Tensor A must be 4D");
    TORCH_CHECK(B.dim() == 2, "Tensor B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch: A.size(3) must equal B.size(0)");

    int BATCH = A.size(0);
    int I = A.size(1);
    int J = A.size(2);
    int L = A.size(3);
    int K = B.size(1);

    auto C = torch::zeros({BATCH, I, J, K}, A.options());

    // Each block computes one element of C. Total blocks = BATCH * I * J * K.
    int totalBlocks = BATCH * I * J * K;
    int threads = 256;

    einsum_kernel_warp_reduce<<<totalBlocks, threads>>>(
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
