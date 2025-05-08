#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void einsum_kernel_unroll(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    extern __shared__ float shared_B[];

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= BATCH * I * J * K) return;

    int k = global_idx % K;
    int remainder = global_idx / K;
    int j = remainder % J;
    remainder /= J;
    int i = remainder % I;
    int b = remainder / I;

    float sum = 0.0f;
    // Process L dimension in tiles with an unroll factor of 4
    for (int l_tile = 0; l_tile < L; l_tile += 4) {
        // Each tile loads 4 rows of B (size: 4*K) into shared memory
        for (int idx = threadIdx.x; idx < 4 * K; idx += blockDim.x) {
            int row = idx / K;
            int col = idx % K;
            int l_index = l_tile + row;
            if (l_index < L)
                shared_B[idx] = B[l_index * K + col];
            else
                shared_B[idx] = 0.0f;
        }
        __syncthreads();

        int a_offset = b * I * J * L + i * J * L + j * L + l_tile;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (l_tile + u < L) {
                sum += A[a_offset + u] * shared_B[u * K + k];
            }
        }
        __syncthreads();
    }

    C[global_idx] = sum;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in l");

    int BATCH = A.size(0), I = A.size(1), J = A.size(2), L = A.size(3);
    int K = B.size(1);
    
    auto C = torch::zeros({BATCH, I, J, K}, A.options());
    int total_elements = BATCH * I * J * K;
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    einsum_kernel_unroll<<<blocks, threads, K * sizeof(float)>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with loop unrolling (CUDA)");}
