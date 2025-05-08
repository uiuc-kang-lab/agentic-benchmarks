#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void hinge_loss_strided_kernel(const float* __restrict__ predictions,
                                        const float* __restrict__ targets,
                                        float* __restrict__ output,
                                        const int n) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements with stride pattern
    for (int idx = blockIdx.x * blockDim.x + tid; idx < n; idx += stride) {
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

    // Optimal thread block size for H100
    const int threads_per_block = 256;
    
    // Calculate optimal number of blocks based on GPU properties
    // Using fewer blocks but letting each thread handle multiple elements
    const int max_blocks = 128;  // Reduced number of blocks since each thread handles multiple elements
    const int num_blocks = min((n + threads_per_block - 1) / threads_per_block, max_blocks);

    hinge_loss_strided_kernel<<<num_blocks, threads_per_block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided Hinge Loss Forward");
}