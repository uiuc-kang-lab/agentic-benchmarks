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
    const int tid = threadIdx.x;
    const int wid = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;
    
    // Calculate base index for the block
    const int block_offset = blockIdx.x * blockDim.x * 4;
    
    // Process 4 elements per thread in a coalesced manner
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    
    // Each thread processes elements stride elements apart
    // This ensures coalesced access within each warp
    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
        const int idx = block_offset / 4 + tid + i * blockDim.x;
        if (idx < vec4_elements) {
            float4 in4 = input4[idx];
            output4[idx] = tanh_vec4<scalar_t>(in4);
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    // Ensure alignment to 128 bytes (32 float4s per warp)
    const int threads = 128; // 4 warps per block
    const int vec4_elements = input.numel() / 4;
    const int blocks = (vec4_elements + (threads * 4) - 1) / (threads * 4);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "coalesced_tanh_kernel", ([&] {
        coalesced_tanh_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            vec4_elements
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Tanh forward (CUDA)");
}