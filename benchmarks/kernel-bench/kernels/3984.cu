#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_shared_kernel(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int num_elements) {
    extern __shared__ float shared_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    if (idx < num_elements) {
        shared_data[tid] = x[idx];
    }
    __syncthreads();

    // Compute softsign using shared memory
    if (idx < num_elements) {
        float val = shared_data[tid];
        out[idx] = val * __fdividef(1.0f, 1.0f + fabsf(val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int num_elements = x.numel();

    const int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    // Launch kernel with shared memory allocation
    softsign_shared_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        num_elements
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign with shared memory optimization (CUDA)");
}