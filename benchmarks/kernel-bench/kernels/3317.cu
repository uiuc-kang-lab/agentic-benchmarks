#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void ldg_swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    // Create thread block group
    cg::thread_block block = cg::this_thread_block();
    
    // Shared memory for loading input values
    extern __shared__ float shared_x[];
    
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        // Cooperatively load data into shared memory
        shared_x[tid] = __ldg(&x[i]);
        block.sync();
        
        // Process data from shared memory
        float val = shared_x[tid];
        // Use fast intrinsic for exp
        float sigmoid = __fdividef(1.0f, 1.0f + __expf(-val));
        y[i] = val * sigmoid;
        
        block.sync();
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    ldg_swish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward with __ldg__ (CUDA)");
}