#include <torch/extension.h>

__global__ void swish_unrolled(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements per iteration
    int n4 = n / 4 * 4;
    for (; idx < n4; idx += 4) {
        float val1 = x[idx];
        float val2 = x[idx + 1];
        float val3 = x[idx + 2];
        float val4 = x[idx + 3];
        
        float sig1 = 1.0f / (1.0f + expf(-val1));
        float sig2 = 1.0f / (1.0f + expf(-val2));
        float sig3 = 1.0f / (1.0f + expf(-val3));
        float sig4 = 1.0f / (1.0f + expf(-val4));
        
        y[idx] = val1 * sig1;
        y[idx + 1] = val2 * sig2;
        y[idx + 2] = val3 * sig3;
        y[idx + 3] = val4 * sig4;
    }
    
    // Handle remaining elements
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
    
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 576);  // 4 blocks/SM * 144 SMs on H100

    swish_unrolled<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish with manual loop unrolling");
}