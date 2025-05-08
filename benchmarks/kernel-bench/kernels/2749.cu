#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <int VECTOR_WIDTH>
__global__ void leaky_relu_vector_kernel(const float* __restrict__ x, float* __restrict__ out, float negative_slope, int n) {
    typedef typename cudaVectorType<float, VECTOR_WIDTH>::type VecType;
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VECTOR_WIDTH;
    if (idx + VECTOR_WIDTH <= n) {
        VecType vec = reinterpret_cast<const VecType*>(x + idx)[0];
        float vals[VECTOR_WIDTH];
        cudaDecomposeVector(vec, vals);
        
        #pragma unroll
        for (int i = 0; i < VECTOR_WIDTH; ++i) {
            vals[i] = fmaxf(vals[i], vals[i] * negative_slope);
        }
        
        reinterpret_cast<VecType*>(out + idx)[0] = cudaComposeVector<float, VECTOR_WIDTH>(vals);
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    constexpr int VECTOR_WIDTH = 4;
    const int threads = 1024;
    const int blocks = (n + threads * VECTOR_WIDTH - 1) / (threads * VECTOR_WIDTH);
    
    leaky_relu_vector_kernel<VECTOR_WIDTH><<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        negative_slope,
        n
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "Vectorized LeakyReLU");
}