#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    const int elements_per_thread = 8;
    int tid = blockIdx.x * blockDim.x * elements_per_thread + threadIdx.x;
    
    float thread_sum = 0.0f;
    
    #pragma unroll
    for(int i = 0; i < elements_per_thread; ++i) {
        int idx = tid + i * blockDim.x;
        if(idx < n_elements) {
            float diff = predictions[idx] - targets[idx];
            float abs_diff = fabsf(diff);
            thread_sum += (abs_diff < 1.0f) ? (0.5f * diff * diff) : (abs_diff - 0.5f);
        }
    }

    // Warp-level reduction
    for(int offset = 16; offset > 0; offset >>= 1)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);

    if((threadIdx.x & 31) == 0) {
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
    const int grid_size = (n + block_size * 8 - 1) / (block_size * 8);

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