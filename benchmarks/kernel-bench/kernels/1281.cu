#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void einsum_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    extern __shared__ float shared_sum[];
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= BATCH * I * J * K) return;

    int k = global_idx % K;
    int remainder = global_idx / K;
    int j = remainder % J;
    remainder /= J;
    int i = remainder % I;
    int b = remainder / I;

    shared_sum[tid] = 0.0f;
    for(int l = tid; l < L; l += blockDim.x) {
        int a_offset = b * I*J*L + i*J*L + j*L + l;
        int b_offset = l*K + k;
        if (l < L) { shared_sum[tid] += A[a_offset] * B[b_offset]; }
    }
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if(tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0) {
        int c_idx = b * I*J*K + i*J*K + j*K + k;
        atomicAdd(&C[c_idx], shared_sum[0]);
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
    int total_elements = BATCH * I * J * K;
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    int shared_mem_size = threads * sizeof(float);
    
    einsum_kernel<<<blocks, threads, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with minimized atomic operations (CUDA)");
}