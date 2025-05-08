#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                              scalar_t* __restrict__ output,
                              int64_t size) {
    constexpr int vec_size = sizeof(float4)/sizeof(scalar_t);
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    if constexpr (std::is_same<scalar_t, float>::value) {
        // Vectorized path for floats with 128-bit alignment
        float4* output_vec = reinterpret_cast<float4*>(output);
        for(int vidx = gidx; vidx*vec_size < size; vidx += stride) {
            const float4* data = reinterpret_cast<const float4*>(input) + vidx;
            float4 val = __ldg(data);  // Cache optimized load
            val.x = 1.0f/(1.0f + __expf(-val.x));
            val.y = 1.0f/(1.0f + __expf(-val.y));
            val.z = 1.0f/(1.0f + __expf(-val.z));
            val.w = 1.0f/(1.0f + __expf(-val.w));
            output_vec[vidx] = val;
        }
    } else {
        // Scalar path for other types
        for(int i = gidx; i < size; i += stride) {
            scalar_t val = __ldg(input + i);
            output[i] = 1.0/(1.0 + exp(-val));
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int64_t size = input.numel();
    
    const int threads = 256;
    int blocks = std::min(65535, (int)((size + threads * 4 - 1)/(threads * 4)));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
        if (std::is_same<scalar_t, float>::value) 
            blocks = (size + (threads * 4) - 1) / (threads * 4);
        else 
            blocks = (size + threads - 1)/threads;
        
        sigmoid_kernel<scalar_t><<<blocks, threads>>>(input.data_ptr<scalar_t>(),
                                                 output.data_ptr<scalar_t>(),
                                                 size);
    });
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized Sigmoid using LDG");
}