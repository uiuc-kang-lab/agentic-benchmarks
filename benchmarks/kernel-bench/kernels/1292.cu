#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_J 32
#define TILE_K 32
#define TILE_L 32

__global__ void einsum_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    __shared__ float As[TILE_J][TILE_L];
    __shared__ float Bs[TILE_L][TILE_K];

    int b = blockIdx.z / I;
    int i = blockIdx.z % I;
    int tj = blockIdx.y * TILE_J;
    int tk = blockIdx.x * TILE_K;

    int j = tj + threadIdx.y;
    int k = tk + threadIdx.x;
    
    float sum = 0.0f;

    for (int tl = 0; tl < L; tl += TILE_L) {
        // Load A tile
        int lA = tl + threadIdx.x;
        if (j < J && lA < L)
            As[threadIdx.y][threadIdx.x] = A[b*I*J*L + i*J*L + j*L + lA];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile
        int lB = tl + threadIdx.y;
        if (lB < L && k < K)
            Bs[threadIdx.y][threadIdx.x] = B[lB*K + k];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial sum
        for (int l = 0; l < TILE_L; l++) {
            sum += As[threadIdx.y][l] * Bs[l][threadIdx.x];
        }

        __syncthreads();
    }

    if (j < J && k < K)
        C[b*I*J*K + i*J*K + j*K + k] = sum;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in l");

    int BATCH = A.size(0), I = A.size(1), J = A.size(2), L = A.size(3);
    int K = B.size(1);

    auto C = torch::zeros({BATCH, I, J, K}, A.options());

    dim3 threads(TILE_K, TILE_J, 1);
    dim3 blocks(
        (K + TILE_K - 1) / TILE_K,
        (J + TILE_J - 1) / TILE_J,
        BATCH * I
    );

    einsum_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), BATCH, I, J, L, K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication (CUDA)");
}