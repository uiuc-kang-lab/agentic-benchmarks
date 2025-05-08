#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void leaky_relu_optimized_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    float negative_slope,
    int n) {
    
    int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(; idx < n; idx += stride) {
        float val = x[idx];
        out[idx] = fmaxf(val, val * negative_slope);
    }
}

torch::Tensor leaky_relu_forward_optimized(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();

    int device_id = x.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    int min_grid, block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size,
        (void*)leaky_relu_optimized_kernel, 0, 0);

    int grid_size = (n + block_size - 1) / block_size;
    grid_size = fmin(grid_size, prop.multiProcessorCount * 32);

    leaky_relu_optimized_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        negative_slope,
        n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_optimized, "LeakyReLU forward optimized (CUDA)");
}