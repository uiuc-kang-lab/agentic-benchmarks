#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_K 32
#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD 4

__global__ void einsum_kernel_adaptive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int IJP,  // Total number of (batch * I * J) groups
    int L,
    int K
) {
    int tid = threadIdx.x;
    int p = blockIdx.x;
    int k_base = blockIdx.y * TILE_K;
    int k = k_base + tid;

    if (p < IJP && k < K) {
        float sum = 0.0f;
        int offsetA = p * L;
        int offsetC = p * K;

        for (int l = 0; l < L; ++l) {
            float a_val = A[offsetA + l];
            float b_val = B[l * K + k];
            sum += a_val * b_val;
        }

        C[offsetC + k] = sum;
    }

    if (blockIdx.y * TILE_K + BLOCK_SIZE * ITEMS_PER_THREAD < K) {
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int k_strided = k_base + tid + i * BLOCK_SIZE;
            if (p < IJP && k_strided < K) {
                float sum_strided = 0.0f;
                int offsetA = p * L;
                int offsetC = p * K;
                for (int l = 0; l < L; ++l) {
                    float a_val = A[offsetA + l];
                    float b_val = B[l * K + k_strided];
                    sum_strided += a_val * b_val;
                }
                C[offsetC + k_strided] = sum_strided;
            }
        }
    }
}

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
    
    int IJP = BATCH * I * J;
    
    int k_blocks = (K + TILE_K - 1) / TILE_K;
    dim3 blocks(IJP, k_blocks);
    dim3 threads(BLOCK_SIZE);

    einsum_kernel_adaptive<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        IJP, L, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with adaptive coalesced and strided processing (CUDA)");
}
