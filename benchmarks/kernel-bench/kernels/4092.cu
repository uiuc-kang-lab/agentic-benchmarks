#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<typename scalar_t>
__global__ void elu_align4_kernel(const scalar_t* __restrict__ x,
                                  scalar_t* __restrict__ out,
                                  float alpha,
                                  int n) {
    const int base_idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = 4 * gridDim.x * blockDim.x;
    
    for(int idx = base_idx; idx < n; idx += stride) {
        // Process full vector4 with boundary protection using min()
        const int remaining = n - idx;
        const int process_size = remaining >= 4 ? 4 : remaining;
        
        float4 vec;
        if (process_size == 4) 
            vec = __ldg(reinterpret_cast<const float4*>(x + idx));
        else {
            const scalar_t* x_ptr = x + idx;
            vec.x = __ldg(x_ptr + 0);
            vec.y = (process_size > 1) ? __ldg(x_ptr + 1) : 0;
            vec.z = (process_size > 2) ? __ldg(x_ptr + 2) : 0;
            vec.w = 0;
        }

        float4 result;
        result.x = (vec.x > 0) ? vec.x : alpha * (expf(vec.x) - 1.0f);
        result.y = (vec.y > 0) ? vec.y : alpha * (expf(vec.y) - 1);
        result.z = (vec.z > 0) ? vec.z : alpha * (expf(vec.z) - 1);

        if (process_size == 4) {
            *reinterpret_cast<float4*>(out + idx) = result;
        } else {
            scalar_t* out_ptr = out + idx;
            out_ptr[0] = result.x;
            if (process_size > 1) out_ptr[1] = result.y;
            if (process_size > 2) out_ptr[2] = result.z;
        }
    }
}

torch::Tensor elu_align4_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    const int n = x.numel();

    constexpr int threads = 256;
    const int max_blocks = 2048;
    const int blocks = max_blocks < (n + 4*threads - 1)/(4*threads) ? max_blocks : (n + 4*threads - 1)/(4*threads);

    elu_align4_kernel<float><<<blocks, threads>>>(x.data_ptr<float>(), 
                                                 out.data_ptr<float>(), 
                                                 alpha, 
                                                 n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_align4_cuda, "Aligned Vector4 ELU (CUDA)");
}
