#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<int ITEMS_PER_THREAD = 4>
__global__ void hinge_loss_optimized_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements
    #pragma unroll
    for (int i = tid; i < n / ITEMS_PER_THREAD; i += stride) {
        float4 pred4, targ4, out4;
        
        // Load 4 elements at once using vectorized loads
        pred4 = *reinterpret_cast<const float4*>(predictions + i * ITEMS_PER_THREAD);
        targ4 = *reinterpret_cast<const float4*>(targets + i * ITEMS_PER_THREAD);
        
        // Process elements
        out4.x = fmaxf(0.0f, 1.0f - pred4.x * targ4.x);
        out4.y = fmaxf(0.0f, 1.0f - pred4.y * targ4.y);
        out4.z = fmaxf(0.0f, 1.0f - pred4.z * targ4.z);
        out4.w = fmaxf(0.0f, 1.0f - pred4.w * targ4.w);
        
        // Store result
        *reinterpret_cast<float4*>(output + i * ITEMS_PER_THREAD) = out4;
    }
    
    // Handle remaining elements
    const int remain_start = (n / ITEMS_PER_THREAD) * ITEMS_PER_THREAD;
    for (int idx = remain_start + tid; idx < n; idx += stride) {
        output[idx] = fmaxf(0.0f, 1.0f - predictions[idx] * targets[idx]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    const int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    // Optimal thread/block configuration
    const int threads = 256;
    const int max_blocks = 65535;
    const int blocks = min((n + threads - 1) / threads, max_blocks);

    hinge_loss_optimized_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Hinge Loss Forward");
}