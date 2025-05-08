#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding) {

    const int out_h = (height + 2 * padding - kernel_h) / stride + 1;
    const int out_w = (width + 2 * padding - kernel_w) / stride + 1;
    
    // Warp-based reduction setup
    const int warp_size = 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;

    // Each warp handles one output element
    const int total_warps = (batch_size * out_channels * out_h * out_w);
    if (warp_id >= total_warps) return;

    // Decode output coordinates from warp ID
    const int oc = warp_id / (batch_size * out_h * out_w);
    const int rem = warp_id % (batch_size * out_h * out_w);
    const int b = rem / (out_h * out_w);
    const int h = (rem % (out_h * out_w)) / out_w;
    const int w = rem % out_w;

    // Calculate kernel bounds
    const int h_start = max(0, -h * stride + padding);
    const int h_end = min(kernel_h, height - h * stride + padding);
    const int w_start = max(0, -w * stride + padding);
    const int w_end = min(kernel_w, width - w * stride + padding);

    // Split input channels across warp threads
    float sum = 0.0f;
    for (int ic = lane_id; ic < in_channels; ic += warp_size) {
        for (int kh = h_start; kh < h_end; ++kh) {
            const int h_in = h * stride + kh - padding;
            for (int kw = w_start; kw < w_end; ++kw) {
                const int w_in = w * stride + kw - padding;
                
                const float input_val = __ldg(&input[((b * in_channels + ic) * height + h_in) * width + w_in]);
                const float weight_val = __ldg(&weight[((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw]);
                sum += input_val * weight_val;
            }
        }
    }

    // Warp-level reduction
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Write final result
    if (lane_id == 0) {
        output[((b * out_channels + oc) * out_h + h) * out_w + w] = sum;
    }
}

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

    if (dilation != 1 || groups != 1) {
        return torch::conv2d(x, weight, bias, {stride, stride},
                           {padding, padding}, {dilation, dilation}, groups);
    }

    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    auto out_h = (height + 2 * padding - kernel_h) / stride + 1;
    auto out_w = (width + 2 * padding - kernel_w) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());

    // Launch configuration with warps as main units
    const int total_warps = batch_size * out_channels * out_h * out_w;
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    conv2d_kernel<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_h,
        kernel_w,
        stride,
        padding
    );

    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution");
}