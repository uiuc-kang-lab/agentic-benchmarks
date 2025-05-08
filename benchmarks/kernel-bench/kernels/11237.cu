#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    // Use registers efficiently by declaring variables only when needed
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    // Process multiple elements per thread to increase arithmetic intensity
    #pragma unroll 4
    for (int i = idx; i < n_elements; i += stride) {
        const float diff = predictions[i] - targets[i];
        const float abs_diff = fabsf(diff);
        // Use branch-free arithmetic to reduce divergent execution
        const bool is_small = abs_diff < 1.0f;
        thread_sum += is_small ? (0.5f * diff * diff) : (abs_diff - 0.5f);
    }

    // Warp reduction using more efficient butterfly pattern
    unsigned int mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // First thread in each warp writes to global memory
    if ((tid & 31) == 0) {  // Faster way to check if thread is first in warp
        atomicAdd(output, thread_sum / n_elements);
    }
}

torch::Tensor smooth_l1_loss_cuda(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(
        predictions.sizes() == targets.sizes(),
        "Input tensors must have the same shape"
    );
    TORCH_CHECK(
        predictions.is_contiguous() && targets.is_contiguous(),
        "Input tensors must be contiguous"
    );
    TORCH_CHECK(
        predictions.device().is_cuda() && targets.device().is_cuda(),
        "Inputs must be CUDA tensors"
    );

    int n = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    smooth_l1_loss_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda, "Smooth L1 Loss (CUDA)");
}