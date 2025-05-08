#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float gelu_scalar(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    const float x_cubed = x * x * x;
    const float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float4 gelu_vectorized(const float4 v) {
    return make_float4(
        gelu_scalar(v.x),
        gelu_scalar(v.y),
        gelu_scalar(v.z),
        gelu_scalar(v.w)
    );
}

__global__ void gelu_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           const int num_elements) {
    const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_vec_threads = gridDim.x * blockDim.x;

    // Process aligned vector chunks
    for (int pos = global_idx; pos < num_elements; pos += vec_stride) {
        if (pos + 3 < num_elements) {
            const float4 vec_in = *reinterpret_cast<const float4*>(input + pos);
            *reinterpret_cast<float4*>(output + pos) = gelu_vectorized(vec_in);
        }
    }

    // Process remaining elements
    const int remainder_start = (num_elements / 4) * 4;
    for (int pos = remainder_start + global_idx; pos < num_elements; pos += total_threads) {
        output[pos] = gelu_scalar(input[pos]);
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "Input must be contiguous CUDA tensor");
    
    auto y = torch::empty_like(x);
    const int num_elements = x.numel();

    // H100-optimized parameters
    constexpr int threads_per_block = 128;
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int blocks = sm_count * 64;  // Utilize all SM partitions
    
    gelu_kernel<<<blocks, threads_per_block>>>(x.data_ptr<float>(),
                                             y.data_ptr<float>(),
                                             num_elements);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Optimized GELU forward");
}