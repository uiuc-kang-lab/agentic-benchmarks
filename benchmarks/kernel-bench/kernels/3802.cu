#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(scalar_t x) {
    // Branchless implementation using smooth max approximation
    const scalar_t pos_thresh = 20.0;
    const scalar_t neg_thresh = -20.0;
    
    // For x > 20, return x
    // For x < -20, return exp(x)
    // Otherwise, return log1p(exp(x))
    scalar_t exp_x = exp(x);
    scalar_t log1p_exp = log1p(exp_x);
    
    // Smooth transition between regions using native CUDA intrinsics
    scalar_t pos_mask = __frcp_rn(1.0f + exp(pos_thresh - x)); // Approaches 1 when x > 20
    scalar_t neg_mask = __frcp_rn(1.0f + exp(x - neg_thresh)); // Approaches 1 when x < -20
    
    return pos_mask * x +                    // x when x > 20
           neg_mask * exp_x +                // exp(x) when x < -20
           (1 - pos_mask - neg_mask) * log1p_exp; // log1p(exp(x)) otherwise
}

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    // Process 4 elements at a time using vector loads/stores when possible
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_size = size / 4;
    
    // Handle vector loads first
    int vec_idx = tid;
    while (vec_idx < vec_size) {
        // Load 4 elements at once
        float4 in_vec = *reinterpret_cast<const float4*>(&input[vec_idx * 4]);
        
        // Process each component
        float4 out_vec;
        out_vec.x = compute_softplus(in_vec.x);
        out_vec.y = compute_softplus(in_vec.y);
        out_vec.z = compute_softplus(in_vec.z);
        out_vec.w = compute_softplus(in_vec.w);
        
        // Store 4 elements at once
        *reinterpret_cast<float4*>(&output[vec_idx * 4]) = out_vec;
        
        vec_idx += stride / 4;
    }
    
    // Handle remaining elements
    int idx = vec_size * 4 + tid;
    while (idx < size) {
        const scalar_t x = __ldg(&input[idx]);
        output[idx] = compute_softplus(x);
        idx += stride;
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    // Ensure block size is multiple of warp size for better memory coalescing
    const int threads = 256;
    const int blocks = min(65535, (size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}