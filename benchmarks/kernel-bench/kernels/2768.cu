#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses grid-stride loops to evenly distribute the workload across all threads and blocks.
// It processes the input in vectorized chunks (using float4) and then handles any leftover elements.

__global__ void grid_stride_leaky_relu_kernel(const float* __restrict__ input,
                                               float* __restrict__ output,
                                               float negative_slope,
                                               int n) {
    // Determine number of full float4 groups and remaining elements
    int groupCount = n / 4;  // number of full vectorized groups
    int remainder = n % 4;     // leftover elements

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Cast pointers for vectorized access
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    float4* output_vec = reinterpret_cast<float4*>(output);

    // Process the vectorized groups with a grid-stride loop
    for (int i = tid; i < groupCount; i += stride) {
        float4 val = input_vec[i];
        float4 res;
        res.x = (val.x > 0.0f ? val.x : val.x * negative_slope);
        res.y = (val.y > 0.0f ? val.y : val.y * negative_slope);
        res.z = (val.z > 0.0f ? val.z : val.z * negative_slope);
        res.w = (val.w > 0.0f ? val.w : val.w * negative_slope);
        output_vec[i] = res;
    }

    // Process any remaining elements that didn't fit in a float4 group
    int offset = groupCount * 4;
    for (int i = tid; i < remainder; i += stride) {
        int idx = offset + i;
        float val = input[idx];
        output[idx] = (val > 0.0f ? val : val * negative_slope);
    }
}

// The forward function sets up the kernel launch parameters such that the workload is evenly distributed
// The grid-stride loop within the single kernel reduces the risk of bottlenecks across blocks and threads.

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 128;
    // Calculate number of vectorized groups; ensure at least 1 block is launched
    int groupCount = n / 4;
    int blocks = ((groupCount > 0 ? groupCount : 1) + threads - 1) / threads;

    grid_stride_leaky_relu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "Grid-strided LeakyReLU forward (CUDA) with even workload distribution");
}
