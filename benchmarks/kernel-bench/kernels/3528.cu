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

__global__ void selu_kernel_vectorized(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      size_t numel) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    const size_t vector_stride = stride * 4;
    size_t vector_idx = idx * 4;

    // Process elements in chunks of 4
    for (; vector_idx < (numel & ~3); vector_idx += vector_stride) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[vector_idx >> 2];
        float4 out_vec;

        process_element(in_vec.x, out_vec.x);
        process_element(in_vec.y, out_vec.y);
        process_element(in_vec.z, out_vec.z);
        process_element(in_vec.w, out_vec.w);

        reinterpret_cast<float4*>(output)[vector_idx >> 2] = out_vec;
    }

    // Handle remaining elements
    const size_t remaining_start = numel & ~3;
    for (size_t i = remaining_start + idx; i < numel; i += stride) {
        float result;
        process_element(input[i], result);
        output[i] = result;
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Input must be float32");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 256;
    const int blocks = (numel + threads * 4 - 1) / (threads * 4);

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    selu_kernel_vectorized<<<blocks, threads>>>(input_ptr, output_ptr, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Vectorized Access (CUDA)");
}