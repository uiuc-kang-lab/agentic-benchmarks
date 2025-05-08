#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float gelu_compute(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = (x + coeff * x_cubed) * sqrt_2_over_pi;
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void gelu_kernel_aligned(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  const int n) {
    // Calculate aligned indices for vector loads
    const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_stride = blockDim.x * gridDim.x;
    const int n_vec = n / 4;

    // Process aligned chunks using float4
    for (int i = vec_idx; i < n_vec; i += vec_stride) {
        // Load 4 elements at once using __ldg
        float4 in_vec;
        const float4* in_ptr = reinterpret_cast<const float4*>(input) + i;
        in_vec = *reinterpret_cast<const float4*>(__ldg(reinterpret_cast<const float*>(in_ptr)));

        // Process each component
        float4 out_vec;
        out_vec.x = gelu_compute(in_vec.x);
        out_vec.y = gelu_compute(in_vec.y);
        out_vec.z = gelu_compute(in_vec.z);
        out_vec.w = gelu_compute(in_vec.w);

        // Store result
        reinterpret_cast<float4*>(output)[i] = out_vec;
    }

    // Handle remaining elements
    const int remaining_start = n_vec * 4;
    for (int idx = remaining_start + threadIdx.x + blockIdx.x * blockDim.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        output[idx] = gelu_compute(__ldg(&input[idx]));
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    const int n = x.numel();
    
    // Adjust block and grid size for better occupancy
    const int threads = 256;
    const int blocks = min(65535, (n + threads * 4 - 1) / (threads * 4));
    
    gelu_kernel_aligned<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Aligned GELU forward CUDA implementation");
}