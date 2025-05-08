#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Constant memory declaration for matrix B
__constant__ float const_B[16384]; // 16KB for small matrices
__shared__ float s_B_tile[16][16]; // Shared memory for B

__global__ void optimized_matmul_kernel(const float* A, float* C, int M, int N, int K) {
    __shared__ float s_A_tile[16][16];
    
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < (K + 15) / 16; ++tile) {
        // Load A tile into shared memory
        int a_col = tile * 16 + threadIdx.x;
        if (row < M && a_col < K)
            s_A_tile[threadIdx.y][threadIdx.x] = A[row*K + a_col];
        else
            s_A_tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Access B matrix from constant memory
        for (int k = 0; k < 16; ++k) {
            int b_row = tile * 16 + k;
            if (b_row < K && col < N)
                sum += s_A_tile[threadIdx.y][k] * const_B[b_row*N + col];
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row*N + col] = sum;
}

void constmem_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Check if B fits in constant memory
    if (K*N > 16384) {
        TORCH_CHECK(false, "Matrix B exceeds constant memory capacity");
    }

    // Copy B to constant memory
    cudaMemcpyToSymbol(const_B, B.data_ptr<float>(), K*N*sizeof(float));

    dim3 blocks((N+15)/16, (M+15)/16);
    dim3 threads(16, 16);
    optimized_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0), K = A.size(1), N = B.size(1);
    torch::Tensor C = torch::zeros({M, N}, A.options());
    
    constmem_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Constant memory matrix multiply (CUDA)");
}