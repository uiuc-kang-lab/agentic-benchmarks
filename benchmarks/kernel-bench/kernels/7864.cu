#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv2d_warp_reduce_kernel(
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
    
    // Warp-based processing
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int output_idx = blockIdx.x * blockDim.x / 32 + warp_id;
    
    if (output_idx >= batch_size * out_channels * out_h * out_w) return;

    // Decompose output index
    const int oc = output_idx / (batch_size * out_h * out_w);
    const int rem = output_idx % (batch_size * out_h * out_w);
    const int b = rem / (out_h * out_w);
    const int h = (rem % (out_h * out_w)) / out_w;
    const int w = rem % out_w;

    // Calculate valid kernel bounds
    const int h_start = max(0, -h * stride + padding);
    const int h_end = min(kernel_h, height - h * stride + padding);
    const int w_start = max(0, -w * stride + padding);
    const int w_end = min(kernel_w, width - w * stride + padding);

    float sum = 0.0f;

    // Distributed channel processing across warp
    for (int ic = lane_id; ic < in_channels; ic += 32) {
        for (int kh = h_start; kh < h_end; ++kh) {
            const int h_in = h * stride + kh - padding;
            for (int kw = w_start; kw < w_end; ++kw) {
                const int w_in = w * stride + kw - padding;
                
                const float input_val = __ldg(&input[
                    ((b * in_channels + ic) * height + h_in) * width + w_in]);
                const float weight_val = __ldg(&weight[
                    ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw]);
                
                sum += input_val * weight_val;
            }
        }
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

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
    
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x);
    CHECK_CUDA(weight); CHECK_CONTIGUOUS(weight);

    if (dilation != 1 || groups != 1) {
        return torch::conv2d(x, weight, bias, {stride, stride},
                           {padding, padding}, {dilation, dilation}, groups);
    }

    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto height = x.size(2);
    const auto width = x.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);

    const int out_h = (height + 2 * padding - kernel_h) / stride + 1;
    const int out_w = (width + 2 * padding - kernel_w) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());

    // Launch configuration
    const int warps_per_block = 8;
    const int threads_per_block = warps_per_block * 32;
    const int total_outputs = batch_size * out_channels * out_h * out_w;
    const int blocks = (total_outputs + warps_per_block - 1) / warps_per_block;

    conv2d_warp_reduce_kernel<<<blocks, threads_per_block>>>(
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
        output += bias.value().view({1, out_channels, 1, 1});
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA 2D Convolution");
}