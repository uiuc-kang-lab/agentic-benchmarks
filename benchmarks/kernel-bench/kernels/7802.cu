#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel computes one output element per block using multiple threads to reduce over the convolution sum.
// Each thread computes a partial sum over a strided portion of the reduction domain.
// Warp-level primitives (__shfl_down_sync) perform intra-warp reductions, and then shared memory is used to combine warp sums.
// The final result (with optional bias addition) is written to the output tensor.

__global__ void conv2d_reduction_shared_warp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // may be nullptr if not provided
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_height,
    int kernel_width,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Grid mapping: gridDim.x represents out_width, gridDim.y represents out_height, and gridDim.z combines batch and output channel.
    int ow = blockIdx.x;  // output width index
    int oh = blockIdx.y;  // output height index
    int b_oc = blockIdx.z;  // combined batch and output channel index
    int b = b_oc / out_channels;     // batch index
    int oc = b_oc % out_channels;      // output channel index

    // Boundary check (should be valid by design)
    if (b >= batch_size || oc >= out_channels) return;

    // Determine grouping parameters
    int group_out_channels = out_channels / groups;
    int group = oc / group_out_channels;  // which group this output channel belongs to
    int in_channels_per_group = in_channels / groups;
    // Reduction domain: over input channels (within group) and kernel spatial dimensions
    int R = in_channels_per_group * kernel_height * kernel_width;

    float partial_sum = 0.0f;
    int tid = threadIdx.x;
    int stride_threads = blockDim.x;  // total threads per block

    // Each thread loops over its portion of the reduction domain
    for (int r = tid; r < R; r += stride_threads) {
        int c = r / (kernel_height * kernel_width);
        int rem = r % (kernel_height * kernel_width);
        int kh = rem / kernel_width;
        int kw = rem % kernel_width;

        int input_channel = group * in_channels_per_group + c;
        int in_y = oh * stride - padding + kh * dilation;
        int in_x = ow * stride - padding + kw * dilation;
        float in_val = 0.0f;
        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
            int input_idx = ((b * in_channels + input_channel) * in_height + in_y) * in_width + in_x;
            in_val = __ldg(&input[input_idx]);
        }
        int weight_idx = ((oc * in_channels_per_group + c) * kernel_height + kh) * kernel_width + kw;
        float w_val = __ldg(&weight[weight_idx]);

        partial_sum += in_val * w_val;
    }

    // Perform warp-level reduction within each warp using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }

    // Shared memory to store the per-warp sums
    extern __shared__ float warp_sums[];  // Size = (blockDim.x / warpSize) * sizeof(float)
    int lane = tid & (warpSize - 1);
    int warpId = tid / warpSize;
    if (lane == 0) {
        warp_sums[warpId] = partial_sum;
    }
    __syncthreads();

    // Let the first warp finalize the reduction across warp sums
    float block_sum = 0.0f;
    if (tid < (blockDim.x / warpSize)) {
        block_sum = warp_sums[tid];
    }
    if (tid < warpSize) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (tid == 0) {
            // Optionally add bias
            if (bias != nullptr) {
                block_sum += __ldg(&bias[oc]);
            }
            int out_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
            output[out_idx] = block_sum;
        }
    }
}

// Forward function: sets up grid dimensions and launches the convolution kernel.
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto options = x.options();
    torch::Tensor output = torch::empty({batch_size, out_channels, out_height, out_width}, options);

    // Each block computes one output element.
    // Grid: (out_width, out_height, batch_size * out_channels)
    dim3 grid(out_width, out_height, batch_size * out_channels);
    // Launch with a block size of 256 threads.
    int block_size = 256;
    // Calculate shared memory size: number of warps in block * sizeof(float)
    int numWarps = (block_size + 31) / 32;
    size_t shared_mem_size = numWarps * sizeof(float);

    conv2d_reduction_shared_warp_kernel<<<grid, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride,
        padding,
        dilation,
        groups
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution with Shared Memory Reductions and Warp-level Primitives");
}
