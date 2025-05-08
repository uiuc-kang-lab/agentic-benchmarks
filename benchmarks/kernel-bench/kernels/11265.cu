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

    // Use efficient warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Each warp's leader writes its sum to shared memory (32 threads per warp)
    __shared__ float shared_sums[32];
    int lane = threadIdx.x % warpSize;
    int warp_idx = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_sums[warp_idx] = thread_sum;
    }
    __syncthreads();

    // Thread 0 aggregates the sums from each warp and performs an atomic add
    if (warp_idx == 0 && lane == 0) {
        float block_sum = 0.0f;
        for (int i = 0; i < blockDim.x / warpSize; i++) {
            block_sum += shared_sums[i];
        }
        atomicAdd(output, block_sum / n_elements);
    }
}

torch::Tensor smooth_l1_loss_cuda(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");

    int n = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    const int grid_size = min((n + block_size - 1) / block_size, 65535);

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