#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Templated kernel to allow different block sizes
template <int BLOCK_SIZE>
__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    // Compute partial sum for each thread
    for (int i = idx; i < n_elements; i += stride) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        if (abs_diff < 1.0f) {
            thread_sum += 0.5f * diff * diff;
        } else {
            thread_sum += abs_diff - 0.5f;
        }
    }

    // Perform block-level reduction using shared memory
    __shared__ float shared_sum[BLOCK_SIZE];
    int tid = threadIdx.x;
    shared_sum[tid] = thread_sum;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    // Accumulate the block sum into the final output
    if (tid == 0) {
        atomicAdd(output, shared_sum[0] / n_elements);
    }
}

// Host function for launching the kernel with a selectable block size
torch::Tensor smooth_l1_loss_cuda(
    torch::Tensor predictions,
    torch::Tensor targets,
    int block_size
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

    // Allowable block sizes for experimentation
    if (block_size != 32 && block_size != 64 && block_size != 128 && block_size != 256 && block_size != 512) {
        throw std::runtime_error("Block size must be one of: 32, 64, 128, 256, or 512");
    }

    int grid_size = (n + block_size - 1) / block_size;

    // Launch the appropriate templated kernel based on the block size
    switch (block_size) {
        case 32:
            smooth_l1_loss_kernel<32><<<grid_size, block_size>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                output.data_ptr<float>(),
                n
            );
            break;
        case 64:
            smooth_l1_loss_kernel<64><<<grid_size, block_size>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                output.data_ptr<float>(),
                n
            );
            break;
        case 128:
            smooth_l1_loss_kernel<128><<<grid_size, block_size>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                output.data_ptr<float>(),
                n
            );
            break;
        case 256:
            smooth_l1_loss_kernel<256><<<grid_size, block_size>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                output.data_ptr<float>(),
                n
            );
            break;
        case 512:
            smooth_l1_loss_kernel<512><<<grid_size, block_size>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                output.data_ptr<float>(),
                n
            );
            break;
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda, "Optimized Smooth L1 Loss (CUDA) with variable block size");
}
