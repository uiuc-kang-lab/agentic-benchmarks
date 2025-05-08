#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized kernel processing float4 and handling remainder
__global__ void leaky_relu_kernel_optimized(const float* x, float* out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;
    int idx4 = idx / 4;
    
    if (idx4 < n4) {
        float4* x4 = (float4*)x;
        float4* out4 = (float4*)out;
        float4 in4 = x4[idx4];
        float4 result;

        result.x = in4.x > 0 ? in4.x : in4.x * negative_slope;
        result.y = in4.y > 0 ? in4.y : in4.y * negative_slope;
        result.z = in4.z > 0 ? in4.z : in4.z * negative_slope;
        result.w = in4.w > 0 ? in4.w : in4.w * negative_slope;

        out4[idx4] = result;
    }
    // Handle remaining elements
    int remainder_start = n4 * 4;
    if (idx >= remainder_start && idx < n) {
        float val = x[idx];
        out[idx] = val > 0 ? val : val * negative_slope;
    }
}

// Host function
torch::Tensor leaky_relu_forward_optimized(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();
    
    // Calculate grid size based on vector width (float4)
    const int vector_width = 4;
    const int threads = 256;
    const int elements_per_thread = vector_width;
    const int elements_per_block = threads * elements_per_thread;
    const int blocks = (n + elements_per_block - 1) / elements_per_block;

    leaky_relu_kernel_optimized<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_optimized, "LeakyReLU forward optimized (CUDA)");
}