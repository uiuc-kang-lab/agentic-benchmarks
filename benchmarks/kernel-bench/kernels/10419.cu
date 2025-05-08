#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float gelu_scalar(float xi) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x3 = xi * xi * xi;
    float inner = sqrt_2_over_pi * (xi + coeff * x3);
    return 0.5f * xi * (1.0f + tanhf(inner));
}

__global__ void gelu_kernel(const float* x, float* y, int n) {
    const int vectors_per_thread = 4; 
    const int stride = blockDim.x * gridDim.x * vectors_per_thread;
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * vectors_per_thread;

    // Process full float4 vectors in aligned chunks
    float4 in, out;
    for(int i = base_idx; i < n/4; i += stride) {
        in = reinterpret_cast<const float4*>(x)[i];
        out.x = gelu_scalar(in.x);
        out.y = gelu_scalar(in.y);
        out.z = gelu_scalar(in.z);
        out.w = gelu_scalar(in.w);
        reinterpret_cast<float4*>(y)[i] = out;
    }

    // Process remaining elements with aligned warp access
    base_idx = base_idx * 4; // Convert vector index to scalar
    for(int i = base_idx; i < n; i += stride*4) {
        if(i + 3 < n) { // Full SIMD width
            float4 tmp;
            tmp.x = gelu_scalar(x[i]);
            tmp.y = gelu_scalar(x[i+1]);
            tmp.z = gelu_scalar(x[i+2]);
            tmp.w = gelu_scalar(x[i+3]);
            *reinterpret_cast<float4*>(&y[i]) = tmp;
        }
        else { // Use volatile stores for tail elements
            for(int j = 0; j < 4 && (i+j) < n; j++)
                y[i+j] = gelu_scalar(x[i+j]);
        }
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "Input must be CUDA contiguous");
    auto y = torch::empty_like(x);
    
    int n = x.numel();
    int num_sm;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);

    const int threads = 128; // 4 warps per block
    int blocks = 4 * num_sm; // Keep SMs busy
    
    gelu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward optimized");
}
