#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float compute_hinge(float pred, float target) {
    return fmaxf(0.0f, 1.0f - pred * target);
}

__global__ void hinge_loss_unrolled_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int unroll_factor = 8;
    
    // Process 8 elements per iteration
    const int aligned_n = n - (n % unroll_factor);
    
    for (int i = tid * unroll_factor; i < aligned_n; i += stride * unroll_factor) {
        float pred[unroll_factor];
        float targ[unroll_factor];
        float result[unroll_factor];
        
        // Manual unroll of loads
        #pragma unroll
        for (int j = 0; j < unroll_factor; j++) {
            pred[j] = __ldg(&predictions[i + j]);
            targ[j] = __ldg(&targets[i + j]);
        }
        
        // Manual unroll of computation
        #pragma unroll
        for (int j = 0; j < unroll_factor; j++) {
            result[j] = compute_hinge(pred[j], targ[j]);
        }
        
        // Manual unroll of stores
        #pragma unroll
        for (int j = 0; j < unroll_factor; j++) {
            output[i + j] = result[j];
        }
    }
    
    // Handle remaining elements
    for (int i = aligned_n + tid; i < n; i += stride) {
        const float pred = __ldg(&predictions[i]);
        const float targ = __ldg(&targets[i]);
        output[i] = compute_hinge(pred, targ);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    const int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    // Optimize thread and block count for unrolled version
    const int threads = 256;
    const int blocks = min((n + threads * 8 - 1) / (threads * 8), 65535);

    hinge_loss_unrolled_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Unrolled Hinge Loss Forward");
}