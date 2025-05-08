#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                             scalar_t* __restrict__ output,
                             const int64_t size) {
    // Calculate warp and lane indices
    const unsigned int warp_size = 32;
    const unsigned int warp_id = threadIdx.x / warp_size;
    const unsigned int lane_id = threadIdx.x % warp_size;
    
    // Calculate the base index for this warp
    const int64_t warp_offset = (blockIdx.x * (blockDim.x / warp_size) + warp_id) * warp_size;
    const int64_t thread_idx = warp_offset + lane_id;
    
    // Check if this warp needs to process data (warp-uniform branch)
    if (warp_offset < size) {
        // All threads in the warp will execute this branch uniformly
        const int64_t remaining = size - warp_offset;
        const int64_t elements_in_warp = min(static_cast<int64_t>(warp_size), remaining);
        
        // Only process if this lane is within the valid range
        // This comparison is uniform within the warp as elements_in_warp is uniform
        if (lane_id < elements_in_warp) {
            float val = static_cast<float>(input[thread_idx]);
            float exp_val = __expf(-val);  // Using fast math intrinsic
            float result = __fdividef(1.0f, (1.0f + exp_val));  // Using fast math intrinsic
            output[thread_idx] = static_cast<scalar_t>(result);
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    
    // Use multiple of warp size for block dimension
    constexpr int threads_per_block = 256;  // 8 warps per block
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();
        
        sigmoid_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input_data, output_data, size);
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward (CUDA)");
}