#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void softplus_kernel_vector(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_size = size / 4;
    
    // Process 4 elements at a time using float4
    float4 in_vec, out_vec;
    
    for (int idx = tid; idx < vec_size; idx += stride) {
        // Load 4 elements aligned to 128-bits
        const scalar_t* aligned_input = reinterpret_cast<const scalar_t*>(__ldg(reinterpret_cast<const float4*>(input + idx * 4)));
        in_vec = reinterpret_cast<const float4&>(aligned_input[0]);
        
        // Process each component
        out_vec.x = (in_vec.x > 20.0f) ? in_vec.x : 
                    (in_vec.x < -20.0f) ? exp(in_vec.x) : log1p(exp(in_vec.x));
        out_vec.y = (in_vec.y > 20.0f) ? in_vec.y : 
                    (in_vec.y < -20.0f) ? exp(in_vec.y) : log1p(exp(in_vec.y));
        out_vec.z = (in_vec.z > 20.0f) ? in_vec.z : 
                    (in_vec.z < -20.0f) ? exp(in_vec.z) : log1p(exp(in_vec.z));
        out_vec.w = (in_vec.w > 20.0f) ? in_vec.w : 
                    (in_vec.w < -20.0f) ? exp(in_vec.w) : log1p(exp(in_vec.w));
        
        // Store aligned result
        reinterpret_cast<float4*>(output + idx * 4)[0] = out_vec;
    }
    
    // Handle remaining elements
    const int remain_start = vec_size * 4;
    for (int idx = remain_start + tid; idx < size; idx += stride) {
        const scalar_t x = __ldg(&input[idx]);
        output[idx] = (x > 20.0f) ? x : 
                     (x < -20.0f) ? exp(x) : log1p(exp(x));
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads * 4 - 1) / (threads * 4);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
        softplus_kernel_vector<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}