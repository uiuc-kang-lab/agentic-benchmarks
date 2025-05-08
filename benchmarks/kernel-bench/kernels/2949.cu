#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tanh_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    using Vec4 = typename std::conditional<std::is_same<scalar_t, float>::value,
                                         float4, double4>::type;
                                         
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Vector size (number of Vec4 elements)
    const int vec_size = size / 4;
    const Vec4* input_vec = reinterpret_cast<const Vec4*>(input);
    Vec4* output_vec = reinterpret_cast<Vec4*>(output);
    
    // Process 4 elements at a time using Vec4
    for (int i = bid * blockDim.x + tid; i < vec_size; i += stride) {
        Vec4 in = input_vec[i];
        Vec4 out;
        out.x = tanh(in.x);
        out.y = tanh(in.y);
        out.z = tanh(in.z);
        out.w = tanh(in.w);
        output_vec[i] = out;
    }
    
    // Handle remaining elements
    const int rem_start = vec_size * 4;
    for (int i = rem_start + bid * blockDim.x + tid; i < size; i += stride) {
        output[i] = tanh(input[i]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    // Optimize for H100: 142 SMs, targeting high occupancy
    const int threads = 256;
    const int blocks = 142 * 2; // 2 blocks per SM
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel", ([&] {
        tanh_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward (CUDA)");
}