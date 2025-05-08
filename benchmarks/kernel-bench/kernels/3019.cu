#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ inline float4 warp_tanh_optimized(const float4 &val) {
    float4 result;
    result.x = tanhf(val.x);
    result.y = tanhf(val.y);
    result.z = tanhf(val.z);
    result.w = tanhf(val.w);
    return result;
}

__global__ void tanh_warp_vector_kernel(const float *__restrict__ input,
                                       float *__restrict__ output,
                                       const int numel) {
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warp.size();
    const int lane_id = threadIdx.x % warp.size();
    
    constexpr int elements_per_warp = 4;
    const int vec_size = numel / (elements_per_warp * 4);

    for (int i = warp_id; i < vec_size; i += gridDim.x * blockDim.x / warp.size()) {
        float4 vec = *reinterpret_cast<const float4*>(&input[(i * elements_per_warp * 4) + lane_id * 4]);
        float4 transformed = warp_tanh_optimized(vec);
        *reinterpret_cast<float4*>(&output[(i * elements_per_warp * 4) + lane_id * 4]) = transformed;
    }

    int residual_start = vec_size * elements_per_warp * 4;
    int residual_idx = residual_start + warp_id * 4 + lane_id;
    if (residual_idx + 3 < numel) {
        float4 residual_vec = *reinterpret_cast<const float4*>(&input[residual_idx]);
        float4 residual_result = warp_tanh_optimized(residual_vec);
        *reinterpret_cast<float4*>(&output[residual_idx]) = residual_result;
    } else if (residual_idx < numel) {
        output[residual_idx] = tanhf(input[residual_idx]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int numel = input.numel();
    
    if (input.scalar_type() == at::ScalarType::Float) {
        const int threads_per_block = 512;
        const int warps_per_block = threads_per_block / 32;
        const int blocks = (numel + (threads_per_block * 4) - 1) / (threads_per_block * 4);
        
        tanh_warp_vector_kernel<<<blocks, threads_per_block>>>( 
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            numel
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_fallback", [&] {
            const int threads = 256;
            const int blocks = (numel + threads - 1) / threads;
            tanh_fallback<scalar_t><<<blocks, threads>>>( 
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                numel
            );
        });
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Tanh with warp vector operations (CUDA)");
}