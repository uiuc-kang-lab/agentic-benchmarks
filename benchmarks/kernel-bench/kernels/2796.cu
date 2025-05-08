#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for frequently accessed values
__constant__ float c_coef[8] = {
    0.5f, -0.5f, 0.25f, 1.0f, 
    0.0f, 1.0f, -1.0f, 2.0f
};

template <typename scalar_t>
__forceinline__ __device__ float fast_sigmoid(float x) {
    // Use constant memory coefficients for computation
    float x_abs = fabsf(x);
    float x_neg = x < c_coef[4]; // 0.0f
    
    // Fast approximation for middle range
    float num = c_coef[0] * x; // 0.5f * x
    float den = c_coef[3] + fabsf(x * c_coef[2]); // 1.0f + |x * 0.25f|
    float result = num / den + c_coef[0]; // result + 0.5f
    
    // Handle extreme values
    if (x_abs > c_coef[7]) { // > 2.0f
        result = x_neg ? c_coef[4] : c_coef[5]; // 0.0f : 1.0f
    }
    
    return result;
}

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                             scalar_t* __restrict__ output,
                             const int64_t size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread using grid-stride loop
    for (int idx = tid; idx < size; idx += stride) {
        float val = static_cast<float>(input[idx]);
        float result = fast_sigmoid<scalar_t>(val);
        output[idx] = static_cast<scalar_t>(result);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    
    constexpr int THREADS = 256;
    const int BLOCKS = std::min(65535, (int)((size + THREADS - 1) / THREADS));
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();
        
        sigmoid_kernel<scalar_t><<<BLOCKS, THREADS>>>(
            input_data, output_data, size);
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward (CUDA)");
}