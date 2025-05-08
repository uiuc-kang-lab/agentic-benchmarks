#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel partitions the reduction domain for each output element so that multiple blocks
// process disjoint slices of the inner summation. Each block reduces its slice using shared memory
// and then performs a single atomicAdd to accumulate its partial sum into the global output.
// The block handling the first slice also adds the bias exactly once. This minimizes the use
// of global atomic operations to only one per block, reducing contention on high-end GPUs like the H100.

__global__ void conv_transposed_1d_atomic_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,   // may be nullptr
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int R,          // reduction domain size = (in_channels/groups) * kernel_size
    int num_splits  // number of splits = ceil(R / blockDim.x)
) {
    // Total number of output elements (without splitting reduction)
    int total_output = batch_size * out_channels * out_length;
    
    // Each block is assigned to a specific output element and a specific slice (split) of the reduction domain.
    int globalBlockId = blockIdx.x;  // global block id
    int output_index = globalBlockId / num_splits;  // one output element is processed by num_splits blocks
    int split_id = globalBlockId % num_splits;
    if (output_index >= total_output) return;

    // Decode output_index into (n, c_out, out_x)
    int n = output_index / (out_channels * out_length);
    int rem = output_index % (out_channels * out_length);
    int c_out = rem / out_length;
    int out_x = rem % out_length;

    // Determine group and local channel indices
    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;
    int group = c_out / out_channels_per_group;
    int c_out_local = c_out % out_channels_per_group;

    // Calculate the slice of the reduction domain handled by this block.
    // Each block uses blockDim.x threads and processes a contiguous tile of size 'tile'.
    int tile = min(blockDim.x, R); // tile size, ensure tile does not exceed R
    int offset_start = split_id * tile;
    int offset_end = (offset_start + tile < R) ? (offset_start + tile) : R;

    // Each thread computes a partial sum over its assigned indices in the tile.
    float local_sum = 0.0f;
    for (int i = offset_start + threadIdx.x; i < offset_end; i += blockDim.x) {
        int channel_local = i / kernel_size;
        int k = i % kernel_size;
        int in_channel = group * in_channels_per_group + channel_local;
        int shifted = out_x + padding - k;
        if ((shifted % stride) == 0) {
            int in_x = shifted / stride;
            if (in_x >= 0 && in_x < in_length) {
                int input_idx = n * (in_channels * in_length) + in_channel * in_length + in_x;
                int weight_idx = in_channel * (out_channels_per_group * kernel_size) + c_out_local * kernel_size + k;
                local_sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    // Reduce partial sums within the block using shared memory.
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 in the block writes the block's reduced partial sum via atomicAdd into global output.
    if (threadIdx.x == 0) {
        float block_sum = sdata[0];
        // Add bias only once per output element (only in the block processing the first split).
        if (split_id == 0 && bias != nullptr) {
            block_sum += bias[c_out];
        }
        atomicAdd(&output[output_index], block_sum);
    }
}

// Host forward function
// Computes output dimension as:
// out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding

torch::Tensor forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_length = input.size(2);
    int kernel_size = weight.size(2);

    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;
    
    int out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    auto output_tensor = torch::zeros({batch_size, out_channels, out_length}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);
    float* output_ptr = output_tensor.data_ptr<float>();

    // Compute the reduction domain size R = (in_channels / groups) * kernel_size
    int in_channels_per_group = in_channels / groups;
    int R = in_channels_per_group * kernel_size;

    // Choose a fixed block size (e.g., 256 threads) and partition R among blocks.
    int threads = 256;
    int num_splits = (R + threads - 1) / threads;

    // Total number of output elements
    int total_output = batch_size * out_channels * out_length;
    // Launch one block for each (output element, reduction slice) pair.
    int blocks = total_output * num_splits;

    auto stream = at::cuda::getCurrentCUDAStream();
    conv_transposed_1d_atomic_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        out_channels,
        in_length,
        out_length,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        R,
        num_splits
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Atomic Optimized Transposed 1D Convolution forward (CUDA)");
}
