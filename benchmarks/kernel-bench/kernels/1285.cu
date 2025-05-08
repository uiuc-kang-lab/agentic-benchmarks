#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM_X 32
#define TILE_DIM_Y 32

__device__ float compute_sum(const float* __restrict__ A, const float* __restrict__ B, int b, int i, int j, int k, int I, int J, int L, int K) {
    float sum = 0.0f;
    #pragma unroll 4
    for (int l = 0; l < L; ++l) {
        int a_idx = b * (I*J*L) + i * (J*L) + j * L + l;
        int b_idx = l * K + k;
        sum += A[a_idx] * B[b_idx];
    }
    return sum;
}

__global__ void einsum_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int j = by * TILE_DIM_Y + ty;
    const int k = bx * TILE_DIM_X + tx;
    
    if (j >= J || k >= K) return;
    
    for (int b = 0; b < BATCH; ++b) {
        for (int i = 0; i < I; ++i) {
            float sum = compute_sum(A, B, b, i, j, k, I, J, L, K);
            int c_idx = b * (I*J*K) + i * (J*K) + j * K + k;
            C[c_idx] = sum;
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in l");

    int BATCH = A.size(0), I = A.size(1), J = A.size(2), L = A.size(3);
    int K = B.size(1);
    
    auto C = torch::zeros({BATCH, I, J, K}, A.options());
    
    dim3 threads(TILE_DIM_X, TILE_DIM_Y);
    dim3 blocks(
        (K + TILE_DIM_X - 1) / TILE_DIM_X,
        (J + TILE_DIM_Y - 1) / TILE_DIM_Y
    );
    
    einsum_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication (CUDA)");
}