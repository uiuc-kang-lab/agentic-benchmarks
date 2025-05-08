#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void hinge_loss_optimized_indexing_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int n
) {
    int block_offset = blockIdx.x * blockDim.x * 8;
    int thread_id = threadIdx.x;

    float local_output[8];

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = block_offset + thread_id + i * blockDim.x;
        if (idx < n) {
            float pred = __ldg(&predictions[idx]);
            float targ = __ldg(&targets[idx]);
            local_output[i] = fmaxf(0.0f, 1.0f - pred * targ);
        } else {
            local_output[i] = 0.0f;
        }
    }

    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = block_offset + thread_id + i * blockDim.x;
        if (idx < n) {
            output[idx] = local_output[i];
        }
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    const int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    const int threads = 256;
    const int blocks = min((n + threads * 8 - 1) / (threads * 8), 65535);

    hinge_loss_optimized_indexing_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Indexing Hinge Loss Forward");
}