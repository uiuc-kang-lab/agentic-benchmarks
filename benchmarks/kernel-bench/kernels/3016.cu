#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tanh_warp_uniform_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int numel) {
    
    // Calculate vector and warp aligned indices
    const int warp_size = 32;
    const int vec_size = 4;
    const int elements_per_warp = warp_size * vec_size;
    const int warp_id = threadIdx.x / warp_size;
    const int lane_id = threadIdx.x % warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    const int global_warp_idx = blockIdx.x * warps_per_block + warp_id;
    
    // Process complete warps (no divergence)
    const int num_complete_warps = numel / elements_per_warp;
    if (global_warp_idx < num_complete_warps) {
        const int base_idx = global_warp_idx * elements_per_warp + lane_id * vec_size;
        float4 in = reinterpret_cast<const float4*>(input)[base_idx/4];
        float4 out;
        out.x = tanhf(in.x);
        out.y = tanhf(in.y);
        out.z = tanhf(in.z);
        out.w = tanhf(in.w);
        reinterpret_cast<float4*>(output)[base_idx/4] = out;
        return;
    }

    // Process remaining elements with uniform control flow
    const int remaining_base = num_complete_warps * elements_per_warp;
    const int remaining_elements = numel - remaining_base;
    if (remaining_elements > 0 && global_warp_idx == num_complete_warps) {
        const int tid = lane_id;
        const int remaining_aligned = (remaining_elements / 4) * 4;
        
        // Process aligned remainder
        for (int i = tid * 4; i < remaining_aligned; i += warp_size * 4) {
            const int idx = remaining_base + i;
            float4 in = reinterpret_cast<const float4*>(&input[idx])[0];
            float4 out;
            out.x = tanhf(in.x);
            out.y = tanhf(in.y);
            out.z = tanhf(in.z);
            out.w = tanhf(in.w);
            reinterpret_cast<float4*>(&output[idx])[0] = out;
        }
        
        // Process final unaligned elements
        const int final_base = remaining_base + remaining_aligned;
        if (tid < (remaining_elements - remaining_aligned)) {
            output[final_base + tid] = tanhf(input[final_base + tid]);
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int elements_per_block = threads * 4;
    const int blocks = (input.numel() + elements_per_block - 1) / elements_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "tanh_warp_uniform_kernel", ([&] {
        tanh_warp_uniform_kernel<<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp uniform Tanh forward (CUDA)");
}