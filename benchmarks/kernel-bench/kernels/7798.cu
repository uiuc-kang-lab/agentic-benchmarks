#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel using warp-level primitives (__shfl_down_sync) for reduction
// Each warp computes one output pixel by distributing the summation over the input channels and kernel window.
__global__ void conv2d_warp_reduce_kernel(
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

    // Number of input channels per group and output channels per group
    int in_channels_per_group = in_channels / groups;
    int group_out_channels = out_channels / groups;

    // Total number of output pixels: batch_size * out_channels * out_height * out_width
    int total_outputs = batch_size * out_channels * out_height * out_width;

    // Each warp computes one output element
    int warpId = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    if (warpId >= total_outputs) return;

    // Lane index within the warp (0-31)
    int lane = threadIdx.x & 31;  // equivalent to threadIdx.x % 32

    // Map warpId to output indices: b, oc, h, w
    int tmp = warpId;
    int w = tmp % out_width;
    tmp /= out_width;
    int h = tmp % out_height;
    tmp /= out_height;
    int oc = tmp % out_channels;
    tmp /= out_channels;
    int b = tmp;

    // Determine corresponding group based on output channel
    int group = oc / group_out_channels;

    // Total reduction iterations = (in_channels_per_group * kernel_height * kernel_width)
    int R = in_channels_per_group * kernel_height * kernel_width;

    float sum = 0.0f;
    // Distribute the reduction work among the 32 warp lanes
    for (int r = lane; r < R; r += 32) {
        int c = r / (kernel_height * kernel_width);
        int rem = r % (kernel_height * kernel_width);
        int kh = rem / kernel_width;
        int kw = rem % kernel_width;

        int in_y = h * stride - padding + kh * dilation;
        int in_x = w * stride - padding + kw * dilation;
        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
            int input_channel = group * in_channels_per_group + c;
            int input_idx = ((b * in_channels + input_channel) * in_height + in_y) * in_width + in_x;
            int weight_idx = (((oc * in_channels_per_group + c) * kernel_height) + kh) * kernel_width + kw;
            sum += input[input_idx] * weight[weight_idx];
        }
    }

    // Perform warp-level reduction using __shfl_down_sync
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Lane 0 in every warp writes the result
    if (lane == 0) {
        if (bias != nullptr) {
            sum += bias[oc];
        }
        int out_idx = ((b * out_channels + oc) * out_height + h) * out_width + w;
        output[out_idx] = sum;
    }
}

// Forward function sets up the kernel launch with each warp responsible for one output pixel
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
    int out_width  = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto options = x.options();
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, options);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* bias_ptr = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);

    int total_outputs = batch_size * out_channels * out_height * out_width;
    // Each warp computes one output element. Use 128 threads per block (4 warps per block).
    int threads_per_block = 128;
    int warps_per_block = threads_per_block / 32;
    int total_warps = total_outputs;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    conv2d_warp_reduce_kernel<<<blocks, threads_per_block>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
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
    m.def("forward", &forward, "CUDA 2D Convolution using Warp-level Reduction");
}
