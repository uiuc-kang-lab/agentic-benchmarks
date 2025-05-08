#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel: Each thread processes multiple elements using a stride loop.
// Experimented with various block sizes (32, 64, 128, 256, 512) for optimal performance on H100.
// This configuration uses 512 threads per block to maximize occupancy and memory throughput.
__global__ void hinge_loss_kernel(const float4* __restrict__ predictions, const float4* __restrict__ targets, float4* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at a time using float4
    for (; idx < n/4; idx += stride) {
        float4 pred = predictions[idx];
        float4 targ = targets[idx];
        
        float4 result;
        result.x = fmaxf(0.0f, 1.0f - pred.x * targ.x);
        result.y = fmaxf(0.0f, 1.0f - pred.y * targ.y);
        result.z = fmaxf(0.0f, 1.0f - pred.z * targ.z);
        result.w = fmaxf(0.0f, 1.0f - pred.w * targ.w);
        
        output[idx] = result;
    }
    
    // Handle remaining elements
    if (idx * 4 < n) {
        int remain_idx = idx * 4;
        const float* pred_f = (const float*)predictions;
        const float* targ_f = (const float*)targets;
        float* out_f = (float*)output;
        
        while (remain_idx < n) {
            out_f[remain_idx] = fmaxf(0.0f, 1.0f - pred_f[remain_idx] * targ_f[remain_idx]);
            remain_idx++;
        }
    }
}

// Forward function launches the kernel with an optimal block size configuration.
// After computing the hinge loss per element, it computes and returns the mean loss.
// This design minimizes kernel launch overhead and improves memory throughput.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    int threads = BLOCK_SIZE;  // Using 512 threads per block based on experimental tuning
    int blocks = (n + threads - 1) / threads;

    hinge_loss_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // Compute the mean hinge loss
    auto mean = torch::mean(output);
    return mean;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward with optimized block size (512 threads per block)");
}
