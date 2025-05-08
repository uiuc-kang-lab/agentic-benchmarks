#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void hinge_loss_adaptive_kernel(const float* __restrict__ predictions, 
                                         const float* __restrict__ targets, 
                                         float* output, 
                                         int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float pred = __ldg(&predictions[idx]);
        const float target = __ldg(&targets[idx]);
        output[idx] = fmaxf(0.0f, 1.0f - pred * target);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    // Adaptive block size selection based on input size
    int block_size;
    if (n <= 256) {
        block_size = 32;  // Small inputs: minimize overhead
    } else if (n <= 1024) {
        block_size = 64;  // Medium-small inputs
    } else if (n <= 4096) {
        block_size = 128; // Medium inputs
    } else if (n <= 16384) {
        block_size = 256; // Medium-large inputs
    } else {
        block_size = 512; // Large inputs: maximize occupancy
    }

    int blocks = (n + block_size - 1) / block_size;

    // Ensure we don't exceed maximum grid dimensions
    blocks = min(blocks, 65535);

    hinge_loss_adaptive_kernel<<<blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // Compute the mean of the output tensor
    auto mean = torch::mean(output);
    return mean;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive Block Size Hinge Loss Forward");
}