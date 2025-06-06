#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline float huber_element(float pred, float target) {
    float diff = pred - target;
    float abs_diff = fabsf(diff);
    return (abs_diff < 1.0f) ? (0.5f * diff * diff) : (abs_diff - 0.5f);
}

__device__ float warp_reduce(float val) {
    for (int delta = 16; delta > 0; delta >>= 1)
        val += __shfl_down_sync(0xffffffff, val, delta);
    return val;
}

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    const int elements_per_thread = 4;
    int idx = blockIdx.x * blockDim.x * elements_per_thread + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < elements_per_thread; ++i) {
        if (idx < n_elements) {
            thread_sum += huber_element(predictions[idx], targets[idx]);
            idx += blockDim.x;
        }
    }

    float block_sum = warp_reduce(thread_sum);

    __shared__ float shared_sum[32];
    if (threadIdx.x % 32 == 0)
        shared_sum[threadIdx.x / 32] = block_sum;
    __syncthreads();

    if (threadIdx.x < 32) {
        float final_sum = (threadIdx.x < blockDim.x / 32) ? shared_sum[threadIdx.x] : 0.0f;
        final_sum = warp_reduce(final_sum);
        if (threadIdx.x == 0)
            atomicAdd(output, final_sum / n_elements);
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
    const int grid_size = min(65535, (n + block_size * 4 - 1) / (block_size * 4));

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