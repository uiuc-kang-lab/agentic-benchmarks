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
__global__ void coalesced_tanh_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int vec4_elements) {
    
    // Ensure coalesced access within warps
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int warps_per_block = blockDim.x / warpSize;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    
    // Calculate base index for this warp ensuring alignment
    const int warp_offset = global_warp_id * warpSize;
    // Ensure consecutive threads access consecutive memory locations
    const int base_idx = (warp_offset + lane_id) * 4;
    const int stride = gridDim.x * blockDim.x * 4;
    
    // Process 4 elements at a time using aligned vector loads/stores
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    
    // Grid-stride loop maintaining coalesced access pattern
    #pragma unroll 4
    for (int idx = base_idx/4; idx < vec4_elements; idx += stride/4) {
        // Load 4 consecutive float4s per warp
        float4 in4 = input4[idx];
        // Process and store maintaining coalescing
        output4[idx] = tanh_vec4<scalar_t>(in4);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    // Ensure alignment for coalesced memory access
    const int vec4_size = input.numel() / 4;
    const int threads = 128; // 4 warps per block
    // Calculate optimal number of blocks for occupancy
    const int sm_count = 108; // H100 has 108 SMs
    const int warps_per_block = threads / 32;
    const int blocks = std::min(
        65535, 
        sm_count * 32 / warps_per_block // Aim for 32 warps per SM
    );
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "coalesced_tanh_kernel", ([&] {
        coalesced_tanh_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            vec4_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Tanh forward (CUDA)");
}