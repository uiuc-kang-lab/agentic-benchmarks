#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void elu_kernel_gridstride(const float4* input, float4* output, float alpha, int n4) {
    const int grid_stride = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (; idx < n4; idx += grid_stride) {
        float4 in = input[idx];
        float4 out;
        out.x = (in.x > 0) ? in.x : alpha * (expf(in.x) - 1);
        out.y = (in.y > 0) ? in.y : alpha * (expf(in.y) - 1);
        out.z = (in.z > 0) ? in.z : alpha * (expf(in.z) - 1);
        out.w = (in.w > 0) ? in.w : alpha * (expf(in.w) - 1);
        output[idx] = out;
    }
}

__global__ void elu_kernel_remainder(const float* input, float* output, float alpha, int offset, int n) {
    const int grid_stride = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    
    for (; idx < n; idx += grid_stride) {
        float val = input[idx];
        output[idx] = (val > 0) ? val : alpha * (expf(val) - 1);
    }
}

torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    
    int n = x.numel();
    int n4 = n / 4;
    int remainder = n % 4;
    
    const int threads_per_block = 256;
    const int max_blocks = 32768;
    const int min_blocks_per_sm = 2;
    const int num_sms = 132;
    
    int num_blocks = min((n4 + threads_per_block - 1) / threads_per_block,
                        min_blocks_per_sm * num_sms);
    num_blocks = min(num_blocks, max_blocks);
    
    if (n4 > 0) {
        elu_kernel_gridstride<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            alpha,
            n4
        );
    }
    
    if (remainder > 0) {
        int remainder_blocks = min((remainder + threads_per_block - 1) / threads_per_block,
                                 min_blocks_per_sm * num_sms);
        remainder_blocks = min(remainder_blocks, max_blocks);
        
        elu_kernel_remainder<<<remainder_blocks, threads_per_block>>>(
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
    m.def("forward", &elu_cuda, "ELU activation with grid stride (CUDA)");
}