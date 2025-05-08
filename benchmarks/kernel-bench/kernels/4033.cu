#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void elu_kernel_shared(const float* __restrict__ x, float* __restrict__ out, float alpha, int n) {
    // Process 4 elements per thread using float4
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    
    if (idx < n - 3) {
        // Load 4 elements at once using float4
        float4 in_val = *((float4*)&x[idx]);
        float4 out_val;
        
        // Process each component
        out_val.x = (in_val.x > 0) ? in_val.x : alpha * (expf(in_val.x) - 1);
        out_val.y = (in_val.y > 0) ? in_val.y : alpha * (expf(in_val.y) - 1);
        out_val.z = (in_val.z > 0) ? in_val.z : alpha * (expf(in_val.z) - 1);
        out_val.w = (in_val.w > 0) ? in_val.w : alpha * (expf(in_val.w) - 1);
        
        // Store 4 elements at once
        *((float4*)&out[idx]) = out_val;
    } else {
        // Handle remaining elements
        for (int i = 0; i < 4 && idx + i < n; i++) {
            float val = x[idx + i];
            out[idx + i] = (val > 0) ? val : alpha * (expf(val) - 1);
        }
    }
}

torch::Tensor elu_cuda_shared(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    size_t shared_memory_size = threads * sizeof(float);

    elu_kernel_shared<<<blocks, threads, shared_memory_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_shared, "ELU activation with shared memory (CUDA)");
}