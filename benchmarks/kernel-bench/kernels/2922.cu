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
__global__ void tanh_kernel_strided(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int vec4_elements) {
    
    // Calculate grid stride for efficient work distribution
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at a time using float4 with grid stride
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    
    // Grid stride loop - each thread processes multiple chunks
    for (int idx = tid; idx < vec4_elements; idx += grid_stride) {
        float4 in4 = input4[idx];
        output4[idx] = tanh_vec4<scalar_t>(in4);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    // Calculate vector elements (groups of 4 floats)
    const int total_elements = input.numel();
    const int vec4_elements = total_elements / 4;
    
    // Optimize thread and block count for better occupancy
    const int threads_per_block = 256;
    const int max_blocks = 65535;
    
    // Calculate optimal number of blocks based on SM count
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    const int blocks_per_sm = 2048 / threads_per_block;
    const int num_blocks = min(max_blocks, num_sms * blocks_per_sm);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_strided", ([&] {
        tanh_kernel_strided<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            vec4_elements
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward with grid stride (CUDA)");
}