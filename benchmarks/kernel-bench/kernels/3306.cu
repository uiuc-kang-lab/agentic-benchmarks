#include <torch/extension.h>

__global__ void swish_kernel(const float* x, float* y, int64_t n) {
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t index = tid * 4;

    if (index + 3 < n) {
        float4 x4 = reinterpret_cast<const float4*>(x + index)[0];
        float4 y4;
        y4.x = x4.x * (1.0f / (1.0f + expf(-x4.x)));
        y4.y = x4.y * (1.0f / (1.0f + expf(-x4.y)));
        y4.z = x4.z * (1.0f / (1.0f + expf(-x4.z)));
        y4.w = x4.w * (1.0f / (1.0f + expf(-x4.w)));
        reinterpret_cast<float4*>(y + index)[0] = y4;
    } else {
        for (int i = 0; i < 4; ++i) {
            if (index + i < n) {
                float val = x[index + i];
                y[index + i] = val * (1.0f / (1.0f + expf(-val)));
            }
        }
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + 4 * threads - 1) / (4 * threads);
    
    swish_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass (CUDA)");
}