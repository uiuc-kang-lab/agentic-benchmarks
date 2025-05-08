#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ float4 tanh_vec4(float4 val) {
    float4 result;
    result.x = tanhf(val.x);
    result.y = tanhf(val.y);
    result.z = tanhf(val.z);
    result.w = tanhf(val.w);
    return result;
}

template <typename scalar_t>
__global__ void tanh_kernel_vectorized_aligned(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int vec4_elements) {
    
    // Calculate aligned indices for vector loads/stores
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = gridDim.x * blockDim.x;
    
    // Process 4 elements at a time using float4
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    
    // Each thread processes multiple vec4 elements in a grid-stride loop
    for (int idx = tid; idx < vec4_elements; idx += stride) {
        float4 in4 = input4[idx];
        output4[idx] = tanh_vec4<scalar_t>(in4);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    // Ensure alignment to warp size (32 threads)
    const int threads = 256;  // 8 warps per block
    const int vec4_size = input.numel() / 4;
    // Ensure blocks are a multiple of warps for better occupancy
    const int warps_per_sm = 32;  // Typical for modern GPUs
    const int blocks = std::min(
        65535,  // Max blocks limit
        (vec4_size + threads - 1) / threads
    );
    
    // Pad input tensor if needed to ensure alignment
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_vectorized_aligned", ([&] {
        tanh_kernel_vectorized_aligned<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            vec4_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward vectorized aligned (CUDA)");
}