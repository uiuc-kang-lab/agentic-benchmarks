#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* x, float* y, int n) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements_per_load = 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);

    #pragma unroll
    for (int i = tid; i < n/elements_per_load; i += blockDim.x * gridDim.x) {
        float4 tmp_x = x4[i];
        float4 tmp_y;

        tmp_y.x = 0.5f * tmp_x.x * (1.0f + tanhf(sqrt_2_over_pi * fmaf(coeff, tmp_x.x*tmp_x.x*tmp_x.x, tmp_x.x)));
        tmp_y.y = 0.5f * tmp_x.y * (1.0f + tanhf(sqrt_2_over_pi * fmaf(coeff, tmp_x.y*tmp_x.y*tmp_x.y, tmp_x.y)));
        tmp_y.z = 0.5f * tmp_x.z * (1.0f + tanhf(sqrt_2_over_pi * fmaf(coeff, tmp_x.z*tmp_x.z*tmp_x.z, tmp_x.z)));
        tmp_y.w = 0.5f * tmp_x.w * (1.0f + tanhf(sqrt_2_over_pi * fmaf(coeff, tmp_x.w*tmp_x.w*tmp_x.w, tmp_x.w)));
        
        y4[i] = tmp_y;
    }

    // Handle remaining elements by padding to avoid warp divergence
    int remainder_start = (n/elements_per_load)*elements_per_load;
    if (tid < blockDim.x) {  // Only use one block for remainder to reduce divergence
        for (int i = remainder_start + tid; i < ((n + 3)/4)*4; i += blockDim.x) {
            if (i < n) {
                float xi = x[i];
                float inner = sqrt_2_over_pi * fmaf(coeff, xi*xi*xi, xi);
                float tanh_val = tanhf(inner);
                y[i] = 0.5f * xi * (1.0f + tanh_val);
            }
        }
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int blocks = sm_count * 128;

    gelu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Vectorized GELU forward CUDA implementation");
}