#include <torch/extension.h>

__global__ void swish_even_workload(const float* x, float* y, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        float val = x[idx];
        float sigmoid = 1.0f / (1.0f + expf(-val));
        y[idx] = val * sigmoid;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    const int threads = 1024;  // Maximize threads per block for better utilization
    int blocks = min((n + threads - 1) / threads, 144 * 4);  // Limit blocks to match H100 resources

    swish_even_workload<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass (CUDA) with even workload distribution");
}