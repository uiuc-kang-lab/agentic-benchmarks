#include <torch/extension.h>
#include <cuda_fp16.h>

template<typename T>
__device__ inline T swish_activation(T val) {
    return val / (1.0f + expf(-val));
}

__global__ void swish_kernel_vectorized(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      const int64_t num_elements) {
    constexpr int VEC_SIZE = 4;
    const int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_stride = blockDim.x * gridDim.x;
    float4* input_vec = reinterpret_cast<float4*>(const_cast<float*>(input));
    float4* output_vec = reinterpret_cast<float4*>(output);
    const int num_vectors = num_elements / VEC_SIZE;

    for(int i = vec_index; i < num_vectors; i += vec_stride) {
        float4 vec = input_vec[i];
        float4 result;
        result.x = swish_activation(vec.x);
        result.y = swish_activation(vec.y);
        result.z = swish_activation(vec.z);
        result.w = swish_activation(vec.w);
        output_vec[i] = result;
    }

    const int scalar_index = vec_index + num_vectors * VEC_SIZE;
    if (scalar_index < num_elements) {
        const int remaining = num_elements - num_vectors * VEC_SIZE;
        for(int r = 0; r < remaining && (scalar_index + r) < num_elements; r++) {
            output[scalar_index + r] = swish_activation(input[scalar_index + r]);
        }
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    
    auto y = torch::empty_like(x);
    const int64_t num_elements = x.numel();
    
    constexpr int threads = 512;
    const int max_blocks = 512;
    const int blocks = (num_elements + threads - 1) / threads;
    
    swish_kernel_vectorized<<<std::min(blocks, max_blocks), threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        num_elements
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(\"forward\", &swish_forward, \"Vectorized Swish forward (CUDA)\");
}
