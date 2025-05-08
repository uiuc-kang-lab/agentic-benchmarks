#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void elu_kernel_ldg_float4(const float4* x, float4* out, float alpha, int n4) {
    const int grid_stride = gridDim.x * blockDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (; idx < n4; idx += grid_stride) {
        float4 in = __ldg(&x[idx]);
        float4 out_val;
        out_val.x = (in.x > 0) ? in.x : alpha * (expf(in.x) - 1);
        out_val.y = (in.y > 0) ? in.y : alpha * (expf(in.y) - 1);
        out_val.z = (in.z > 0) ? in.z : alpha * (expf(in.z) - 1);
        out_val.w = (in.w > 0) ? in.w : alpha * (expf(in.w) - 1);
        out[idx] = out_val;
    }
}

__global__ void elu_kernel_remainder_ldg(const float* x, float* out, float alpha, int offset, int n) {
    const int grid_stride = gridDim.x * blockDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    
    for (; idx < n; idx += grid_stride) {
        float val = __ldg(&x[idx]);
        out[idx] = (val > 0) ? val : alpha * (expf(val) - 1);
    }
}

torch::Tensor elu_cuda_ldg(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    
    int n = x.numel();
    int n4 = n / 4;
    int remainder = n % 4;
    
    const int threads_per_block = 256;
    const int num_blocks = (n4 + threads_per_block - 1) / threads_per_block;
    
    if (n4 > 0) {
        elu_kernel_ldg_float4<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            alpha,
            n4
        );
    }
    
    if (remainder > 0) {
        const int tail_blocks = (remainder + threads_per_block - 1) / threads_per_block;
        elu_kernel_remainder_ldg<<<tail_blocks, threads_per_block>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            alpha,
            n4 * 4,
            n
        );
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_ldg, "ELU activation with __ldg optimization (CUDA)");
}