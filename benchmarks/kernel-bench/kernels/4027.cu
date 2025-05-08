#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Process 4 float4 elements (16 values) per thread
__device__ __forceinline__ void process_float4(const float4& in, float4& out, float alpha) {
    out.x = (in.x > 0) ? in.x : alpha * (expf(in.x) - 1);
    out.y = (in.y > 0) ? in.y : alpha * (expf(in.y) - 1);
    out.z = (in.z > 0) ? in.z : alpha * (expf(in.z) - 1);
    out.w = (in.w > 0) ? in.w : alpha * (expf(in.w) - 1);
}

__global__ void elu_kernel_unrolled(const float4* input, float4* output, float alpha, int n4) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int items_per_thread = 4;  // Process 4 float4 elements per thread
    
    // Main loop with manual unrolling
    int idx = tid;
    #pragma unroll
    for (int i = 0; i < items_per_thread; i++) {
        if (idx < n4) {
            float4 in = input[idx];
            float4 out;
            process_float4(in, out, alpha);
            output[idx] = out;
            idx += stride;
        }
    }
    
    // Handle remaining elements
    for (; idx < n4; idx += stride) {
        float4 in = input[idx];
        float4 out;
        process_float4(in, out, alpha);
        output[idx] = out;
    }
}

__global__ void elu_kernel_remainder(const float* input, float* output, float alpha, int offset, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    #pragma unroll 4
    for (int idx = tid + offset; idx < n; idx += stride) {
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
    const int num_sms = 132;  // H100 specific
    const int blocks_per_sm = 2;
    int num_blocks = min((n4 + threads_per_block - 1) / threads_per_block,
                        blocks_per_sm * num_sms);
    
    if (n4 > 0) {
        elu_kernel_unrolled<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            alpha,
            n4
        );
    }
    
    if (remainder > 0) {
        int remainder_blocks = min((remainder + threads_per_block - 1) / threads_per_block,
                                 blocks_per_sm * num_sms);
        
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
    m.def("forward", &elu_cuda, "ELU activation with unrolled processing (CUDA)");
}