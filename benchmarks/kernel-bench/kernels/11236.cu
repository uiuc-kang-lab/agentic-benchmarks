#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    for (int i = idx; i < n_elements; i += stride) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? (0.5f * diff * diff) : (abs_diff - 0.5f);
    }

    // Warp-level reduction
    float val = thread_sum;
    for (int delta = 16; delta > 0; delta >>= 1)
        val += __shfl_down_sync(0xffffffff, val, delta);

    // Store warp sums in shared memory
    __shared__ float shared_sum[8];  // 256 threads / 32 warp size = 8 warps
    if (threadIdx.x % 32 == 0)
        shared_sum[threadIdx.x / 32] = val;
    __syncthreads();

    // Final reduction by first warp
    if (threadIdx.x < 8) {
        float warp_sum = shared_sum[threadIdx.x];
        for (int delta = 4; delta > 0; delta >>= 1)
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, delta);
        
        if (threadIdx.x == 0)
            atomicAdd(output, warp_sum / n_elements);
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