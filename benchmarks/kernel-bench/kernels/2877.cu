#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T>
__device__ __forceinline__ T sigmoid_compute(T val) {
    T exp_val = expf(-val);
    return 1.0f / (1.0f + exp_val);
}

template<typename scalar_t>
__global__ void sigmoid_kernel_vectorized(const scalar_t* __restrict__ input,
                                         scalar_t* __restrict__ output,
                                         const int64_t size) {
    constexpr int vec_size = sizeof(float4) / sizeof(scalar_t);
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process multiple vectors per thread using grid-stride loop
    for (int idx = tid; idx < size / vec_size; idx += stride) {
        int i = idx * vec_size;
        if (i < size) {
            float4 in_chunk = *reinterpret_cast<const float4*>(&input[i]);
            float4 out_chunk;
            
            out_chunk.x = sigmoid_compute(in_chunk.x);
            out_chunk.y = sigmoid_compute(in_chunk.y);
            out_chunk.z = sigmoid_compute(in_chunk.z);
            out_chunk.w = sigmoid_compute(in_chunk.w);
            
            *reinterpret_cast<float4*>(&output[i]) = out_chunk;
        }
    }
    
    // Handle remaining elements
    int remainder = size % vec_size;
    int remainder_start = size - remainder;
    if (tid < remainder) {
        output[remainder_start + tid] = sigmoid_compute(input[remainder_start + tid]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();

    const int threads = 256;
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    const int blocks_per_sm = 2;  // Adjust based on occupancy requirements
    const int total_blocks = num_sms * blocks_per_sm;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
        sigmoid_kernel_vectorized<scalar_t><<<total_blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced Vectorized Sigmoid forward (CUDA)");
}