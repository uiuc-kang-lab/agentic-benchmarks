#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device constants for SELU parameters
__device__ constexpr float ALPHA = 1.67326324235437728481f;
__device__ constexpr float SCALE = 1.05070098735548049342f;

// Predicated SELU computation without branching
__device__ __forceinline__ float compute_selu(float x) {
    float exp_val = expf(x);
    float neg_result = ALPHA * (exp_val - 1.0f);
    // Use predication instead of branching
    float result = x * (x > 0.0f) + neg_result * (x <= 0.0f);
    return SCALE * result;
}

__global__ void selu_kernel_warp_uniform(const float4* __restrict__ input,
                                        float4* __restrict__ output,
                                        size_t vector_size) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    // Process vector elements
    for (size_t i = tid; i < vector_size; i += stride) {
        float4 in_vec = __ldg(&input[i]);
        float4 out_vec;
        
        // Process all components without branching
        out_vec.x = compute_selu(in_vec.x);
        out_vec.y = compute_selu(in_vec.y);
        out_vec.z = compute_selu(in_vec.z);
        out_vec.w = compute_selu(in_vec.w);
        
        output[i] = out_vec;
    }
}

// Handles remaining elements without branching
__global__ void selu_kernel_remainder(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     size_t start,
                                     size_t total_size) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = start + tid; i < total_size; i += stride) {
        output[i] = compute_selu(input[i]);
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Input must be float32");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const size_t vector_size = numel / 4;
    const size_t vector_elements = vector_size * 4;
    
    const int threads = 256;
    const int blocks = (vector_size + threads - 1) / threads;

    // Main vectorized processing
    selu_kernel_warp_uniform<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(input.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        vector_size
    );

    // Handle remaining elements
    if (vector_elements < numel) {
        const int remainder_blocks = (numel - vector_elements + threads - 1) / threads;
        selu_kernel_remainder<<<remainder_blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            vector_elements,
            numel
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Uniform Warp Execution (CUDA)");
}