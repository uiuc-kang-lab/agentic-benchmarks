#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void einsum_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= BATCH * I * J * K) return;

    int k = global_idx % K;
    int remainder = global_idx / K;
    int j = remainder % J;
    remainder /= J;
    int i = remainder % I;
    int b = remainder / I;

    float sum = 0.0f;
    for(int l = 0; l < L; ++l) {
        int a_offset = b * I*J*L + i*J*L + j*L + l;
        int b_offset = l*K + k;
        sum += A[a_offset] * B[b_offset];
    }
    
    C[global_idx] = sum;
}

void forward_with_streams(torch::Tensor A, torch::Tensor B, torch::Tensor C, cudaStream_t stream) {
    int BATCH = A.size(0), I = A.size(1), J = A.size(2), L = A.size(3);
    int K = B.size(1);
    
    int total_elements = BATCH * I * J * K;
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    einsum_kernel<<<blocks, threads, 0, stream>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in l");

    int BATCH = A.size(0), I = A.size(1), J = A.size(2), L = A.size(3);
    int K = B.size(1);
    
    auto C = torch::zeros({BATCH, I, J, K}, A.options());
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    forward_with_streams(A, B, C, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with streams (CUDA)");
}