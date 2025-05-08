#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__constant__ float d_alpha;

template<typename scalar_t>
__global__ void elu_const_kernel(const scalar_t* __restrict__ x,
                                scalar_t* __restrict__ out,
                                int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Process 4 elements per thread
    for (int i = tid * 4; i < n; i += stride * 4) {
        if (i + 3 < n) {
            // Vector load
            float4 in = *reinterpret_cast<const float4*>(x + i);
            float4 result;
            
            // Process each component
            result.x = (in.x > 0) ? in.x : d_alpha * (expf(in.x) - 1);
            result.y = (in.y > 0) ? in.y : d_alpha * (expf(in.y) - 1);
            result.z = (in.z > 0) ? in.z : d_alpha * (expf(in.z) - 1);
            result.w = (in.w > 0) ? in.w : d_alpha * (expf(in.w) - 1);
            
            // Vector store
            *reinterpret_cast<float4*>(out + i) = result;
        } else {
            // Handle remaining elements
            for (int j = 0; j < 4 && i + j < n; ++j) {
                float val = x[i + j];
                out[i + j] = (val > 0) ? val : d_alpha * (expf(val) - 1);
            }
        }
    }
}

torch::Tensor elu_const_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    const int n = x.numel();

    // Copy alpha to constant memory
    cudaMemcpyToSymbol(d_alpha, &alpha, sizeof(float));

    const int threads = 256;
    const int blocks = (n + threads * 4 - 1) / (threads * 4);

    elu_const_kernel<float><<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_const_cuda, "Constant memory ELU activation (CUDA)");
}