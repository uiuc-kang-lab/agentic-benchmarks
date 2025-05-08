#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float4 softsign_vec4_compute(float4 input) {
    float4 output;
    output.x = input.x / (1.0f + fabsf(input.x));
    output.y = input.y / (1.0f + fabsf(input.y));
    output.z = input.z / (1.0f + fabsf(input.z));
    output.w = input.w / (1.0f + fabsf(input.w));
    return output;
}

__global__ void softsign_kernel_coalesced(const float* __restrict__ x, 
                                        float* __restrict__ out,
                                        const int num_elements) {
    // Calculate aligned indices for coalesced memory access
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;  // Division by 32 (warp size)
    const int lane_id = tid & 31;  // Modulo 32 (warp size)
    
    // Each thread processes 4 elements at a time using float4
    const int elements_per_thread = 4;
    const int elements_per_block = blockDim.x * elements_per_thread;
    int base_idx = blockIdx.x * elements_per_block + (tid * elements_per_thread);
    
    // Ensure aligned access for float4
    if (base_idx + 3 < num_elements) {
        // Load 4 elements at once (coalesced read)
        float4 input = reinterpret_cast<const float4*>(x)[base_idx >> 2];
        
        // Process the 4 elements
        float4 result = softsign_vec4_compute(input);
        
        // Store 4 elements at once (coalesced write)
        reinterpret_cast<float4*>(out)[base_idx >> 2] = result;
    } else {
        // Handle remaining elements
        for (int i = 0; i < 4 && base_idx + i < num_elements; i++) {
            float val = x[base_idx + i];
            out[base_idx + i] = val / (1.0f + fabsf(val));
        }
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    // Configure launch parameters for optimal memory coalescing
    const int threads = 128;  // Multiple of warp size (32)
    const int elements_per_thread = 4;
    const int elements_per_block = threads * elements_per_thread;
    const int blocks = (num_elements + elements_per_block - 1) / elements_per_block;
    
    softsign_kernel_coalesced<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        num_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Softsign activation (CUDA)");
}