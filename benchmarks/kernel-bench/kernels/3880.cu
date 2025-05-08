#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_kernel_optimized(const float* x, float* out, int num_elements) {
    extern __shared__ float shared_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;

    // Load data into shared memory only if within bounds
    if (idx < num_elements) {
        shared_data[local_idx] = x[idx];
    }
    __syncthreads();

    // Perform the Softsign computation
    if (idx < num_elements) {
        float val = shared_data[local_idx];
        out[idx] = val / (1.0f + fabsf(val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;

    size_t shared_memory_size = threads * sizeof(float);
    softsign_kernel_optimized<<<blocks, threads, shared_memory_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), num_elements
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Softsign activation (CUDA)");
}