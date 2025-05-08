#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_kernel_dp(const float* x, float* out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = x[idx];
        out[idx] = __fdividef(val, 1.0f + fabsf(val));
    }
}

__global__ void distribute_more_jobs(const float* x, float* out, int num_elements) {
    dim3 block(256);
    dim3 grid((num_elements + 255) / 256);
    softsign_kernel_dp<<<grid, block>>>(x, out, num_elements);
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int num_elements = x.numel();

    dim3 block(128);
    dim3 grid((num_elements + 127) / 128);

    distribute_more_jobs<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), num_elements);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign dynamic parallelism (CUDA)");
}