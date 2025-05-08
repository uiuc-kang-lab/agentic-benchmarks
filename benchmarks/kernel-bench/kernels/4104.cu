#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<typename scalar_t>
__global__ void elu_vector8_kernel(const scalar_t* __restrict__ x,
                                  scalar_t* __restrict__ out,
                                  float alpha,
                                  int n) {
    constexpr int VEC_SIZE = 4;
    int tid = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int stride = gridDim.x * blockDim.x * 2;

    #pragma unroll
    for (int base = tid * VEC_SIZE; base < n; base += stride * VEC_SIZE) {
        // Process 2 vectors (8 elements total) per iteration
        if (base + 7 * VEC_SIZE < n) {
            float4 vec1 = *reinterpret_cast<const float4*>(x + base);
            float4 vec2 = *reinterpret_cast<const float4*>(x + base + 4);

            float4 res1, res2;
            res1.x = (vec1.x > 0) ? vec1.x : alpha * (expf(vec1.x) - 1);
            res1.y = (vec1.y > 0) ? vec1.y : alpha * (expf(vec1.y) - 1);
            res1.z = (vec1.z > 0) ? vec1.z : alpha * (expf(vec1.z) - 1);
            res1.w = (vec1.w > 0) ? vec1.w : alpha * (expf(vec1.w) - 1);

            res2.x = (vec2.x > 0) ? vec2.x : alpha * (expf(vec2.x) - 1);
            res2.y = (vec2.y > 0) ? vec2.y : alpha * (expf(vec2.y) - 1);
            res2.z = (vec2.z > 0) ? vec2.z : alpha * (expf(vec2.z) - 1);
            res2.w = (vec2.w > 0) ? vec2.w : alpha * (expf(vec2.w) - 1);

            *reinterpret_cast<float4*>(out + base) = res1;
            *reinterpret_cast<float4*>(out + base + 4) = res2;
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int pos = base + i;
                if (pos < n) {
                    float val = x[pos];
                    out[pos] = (val > 0) ? val : alpha * (expf(val) - 1);
                }
            }
        }
    }
}

torch::Tensor elu_vector8_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    constexpr int threads = 256;
    const int blocks = min((n + threads * 8 - 1) / (threads * 8), 128);

    elu_vector8_kernel<float><<<blocks, threads>>>(x.data_ptr<float>(),
                                                 out.data_ptr<float>(),
                                                 alpha,
                                                 n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_vector8_cuda, "Vectorized ELU with 8-element unrolling (CUDA)");
}
