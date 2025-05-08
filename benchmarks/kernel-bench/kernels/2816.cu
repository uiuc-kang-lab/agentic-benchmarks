#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel processes data at the warp level to minimize divergence
// Each warp processes 32 contiguous elements, and uses __ballot_sync to detect
// if all lanes are active. Fully active warps perform uniform stores without additional conditions,
// while partially active warps only store for valid indices.

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                                 scalar_t* __restrict__ output,
                                 const int64_t size) {
    // Each block is assumed to have a multiple of 32 threads
    const int warps_per_block = blockDim.x / 32;
    const int global_warp_id = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int total_warps = (size + 31) >> 5;  // equivalent to ceil(size/32)

    // Use grid stride loop over warps to cover the entire array
    for (int warp_id = global_warp_id; warp_id < total_warps; warp_id += gridDim.x * warps_per_block) {
        int idx = warp_id * 32 + lane;
        // Obtain a warp-wide mask: if (idx < size) for all threads in the warp
        unsigned int active_mask = __ballot_sync(0xffffffff, idx < size);
        
        // Load input if in-bound; if not, use a dummy value (won't be stored if not active)
        float in_val = (idx < size) ? static_cast<float>(input[idx]) : 0.0f;
        float out_val = 1.0f / (1.0f + expf(-in_val));
        
        // If all threads in the warp are valid, perform the store uniformly
        if (active_mask == 0xffffffff) {
            output[idx] = static_cast<scalar_t>(out_val);
        } else {
            // For partially active warps, only valid threads store the result
            if (idx < size) {
                output[idx] = static_cast<scalar_t>(out_val);
            }
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    
    // Configure kernel launch parameters
    const int threads = 256;  // Must be a multiple of 32 for warp-level processing
    const int warps_per_block = threads / 32;
    const int total_warps = (size + 31) / 32;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", ([&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();
        
        sigmoid_kernel<scalar_t><<<blocks, threads>>>(input_data, output_data, size);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward (CUDA)");
}
