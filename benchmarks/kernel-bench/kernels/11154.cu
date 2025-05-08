#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_loss_kernel_optimized(
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
        if (abs_diff < 1.0f) {
            thread_sum += 0.5f * diff * diff;
        } else {
            thread_sum += abs_diff - 0.5f;
        }
    }

    // Block-wise reduction
    __shared__ float shared_sum[512];  // Adjusting shared memory size for larger block sizes
    int tid = threadIdx.x;
    shared_sum[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared_sum[0] / n_elements);
    }
}

torch::Tensor smooth_l1_loss_cuda_optimized(
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

    const int block_size = 512;  // Experimented and found optimal block size
    const int grid_size = (n + block_size - 1) / block_size;

    smooth_l1_loss_kernel_optimized<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda_optimized, "Smooth L1 Loss Optimized (CUDA)");
}