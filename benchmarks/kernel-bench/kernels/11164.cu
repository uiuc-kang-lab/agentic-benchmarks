#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_loss_kernel_aggressive_unroll(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    // Process 8 elements per thread per iteration
    #pragma unroll
    for (int i = idx; i < n_elements - 7; i += stride * 8) {
        float diff0, diff1, diff2, diff3, diff4, diff5, diff6, diff7;
        float abs_diff0, abs_diff1, abs_diff2, abs_diff3, abs_diff4, abs_diff5, abs_diff6, abs_diff7;
        float partial_sum = 0.0f;

        // Load and process 8 elements
        if (i < n_elements) {
            diff0 = predictions[i] - targets[i];
            abs_diff0 = fabsf(diff0);
            partial_sum += (abs_diff0 < 1.0f) ? 0.5f * diff0 * diff0 : abs_diff0 - 0.5f;
        }
        if (i + stride < n_elements) {
            diff1 = predictions[i + stride] - targets[i + stride];
            abs_diff1 = fabsf(diff1);
            partial_sum += (abs_diff1 < 1.0f) ? 0.5f * diff1 * diff1 : abs_diff1 - 0.5f;
        }
        if (i + 2 * stride < n_elements) {
            diff2 = predictions[i + 2 * stride] - targets[i + 2 * stride];
            abs_diff2 = fabsf(diff2);
            partial_sum += (abs_diff2 < 1.0f) ? 0.5f * diff2 * diff2 : abs_diff2 - 0.5f;
        }
        if (i + 3 * stride < n_elements) {
            diff3 = predictions[i + 3 * stride] - targets[i + 3 * stride];
            abs_diff3 = fabsf(diff3);
            partial_sum += (abs_diff3 < 1.0f) ? 0.5f * diff3 * diff3 : abs_diff3 - 0.5f;
        }
        if (i + 4 * stride < n_elements) {
            diff4 = predictions[i + 4 * stride] - targets[i + 4 * stride];
            abs_diff4 = fabsf(diff4);
            partial_sum += (abs_diff4 < 1.0f) ? 0.5f * diff4 * diff4 : abs_diff4 - 0.5f;
        }
        if (i + 5 * stride < n_elements) {
            diff5 = predictions[i + 5 * stride] - targets[i + 5 * stride];
            abs_diff5 = fabsf(diff5);
            partial_sum += (abs_diff5 < 1.0f) ? 0.5f * diff5 * diff5 : abs_diff5 - 0.5f;
        }
        if (i + 6 * stride < n_elements) {
            diff6 = predictions[i + 6 * stride] - targets[i + 6 * stride];
            abs_diff6 = fabsf(diff6);
            partial_sum += (abs_diff6 < 1.0f) ? 0.5f * diff6 * diff6 : abs_diff6 - 0.5f;
        }
        if (i + 7 * stride < n_elements) {
            diff7 = predictions[i + 7 * stride] - targets[i + 7 * stride];
            abs_diff7 = fabsf(diff7);
            partial_sum += (abs_diff7 < 1.0f) ? 0.5f * diff7 * diff7 : abs_diff7 - 0.5f;
        }
        
        thread_sum += partial_sum;
    }

    // Handle remaining elements
    for (int i = idx + ((n_elements >> 3) << 3); i < n_elements; i += stride) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Two-level reduction with explicit unrolling
    __shared__ float shared_sum[256];
    int tid = threadIdx.x;
    shared_sum[tid] = thread_sum;
    __syncthreads();

    // Explicit unrolling of reduction loop
    if (tid < 128) shared_sum[tid] += shared_sum[tid + 128];
    __syncthreads();
    if (tid < 64) shared_sum[tid] += shared_sum[tid + 64];
    __syncthreads();
    if (tid < 32) {
        shared_sum[tid] += shared_sum[tid + 32];
        shared_sum[tid] += shared_sum[tid + 16];
        shared_sum[tid] += shared_sum[tid + 8];
        shared_sum[tid] += shared_sum[tid + 4];
        shared_sum[tid] += shared_sum[tid + 2];
        shared_sum[tid] += shared_sum[tid + 1];
    }

    if (tid == 0) {
        atomicAdd(output, shared_sum[0] / n_elements);
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

    smooth_l1_loss_kernel_aggressive_unroll<<<grid_size, block_size>>>(
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