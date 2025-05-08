#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: define an inline exponential function for float
__device__ inline float my_exp(float x) {
    return expf(x);
}

__device__ inline void process_element(float x, float& result) {
    result = (x > 0.0f)
        ? x 
        : 1.67326324235437728481f * (my_exp(x) - 1.0f);
    result *= 1.05070098735548049342f;
}

__global__ void selu_kernel_stride_optimized(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      size_t numel) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    // Process elements using stride loop
    for (size_t i = idx; i < numel; i += stride) {
        float x = __ldg(&input[i]);
        float result;
        process_element(x, result);
        output[i] = result;
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Input must be float32");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    selu_kernel_stride_optimized<<<blocks, threads>>>(input_ptr, output_ptr, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Stride Optimization (CUDA)");
}