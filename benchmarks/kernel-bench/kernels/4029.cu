#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void elu_kernel_shared(const float* x, float* out, float alpha, int n) {
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    if (idx < n) {
        sdata[tid] = x[idx];
    }
    __syncthreads();

    // Perform ELU operation using shared memory
    if (idx < n) {
        float val = sdata[tid];
        out[idx] = (val > 0) ? val : alpha * (expf(val) - 1);
    }
}

torch::Tensor elu_cuda_shared(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    size_t shared_memory_size = threads * sizeof(float);

    elu_kernel_shared<<<blocks, threads, shared_memory_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_shared, "ELU activation with shared memory (CUDA)");
}