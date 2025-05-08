#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

using std::max;
using std::min;

// This kernel has a specialized branch for the common 3x3 convolution with stride=1 and padding=1.
// In that case, for interior pixels, the inner loop is manually unrolled to reduce loop overhead.
// For all other cases, the kernel falls back to a generic loop using #pragma unroll to assist with unrolling.

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding
) {
    // Compute output dimensions
    int out_h = (height + 2 * padding - kernel_h) / stride + 1;
    int out_w = (width + 2 * padding - kernel_w) / stride + 1;
    int total = batch_size * out_channels * out_h * out_w;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= total) return;

    // Decode flattened index into (b, oc, h, w)
    int b = index / (out_channels * out_h * out_w);
    int rem = index % (out_channels * out_h * out_w);
    int oc = rem / (out_h * out_w);
    int rem2 = rem % (out_h * out_w);
    int h = rem2 / out_w;
    int w = rem2 % out_w;

    float sum = 0.0f;

    // Specialized manual unrolling for 3x3 kernel with stride 1 and padding 1 for interior pixels
    if(kernel_h == 3 && kernel_w == 3 && stride == 1 && padding == 1 &&
       h >= 1 && h < (out_h - 1) && w >= 1 && w < (out_w - 1)) {
        // For interior pixels, the 3x3 patch is fully applied
        for (int ic = 0; ic < in_channels; ic++) {
            // Compute base offset for the input slice for this channel
            int base = ((b * in_channels + ic) * height) * width;
            // (h, w) in output corresponds to center of a 3x3 patch in input when stride==1, padding==1
            int input_offset = base + (h - 1) * width + (w - 1);
            int weight_base = ((oc * in_channels + ic) * 9); // 3*3 = 9 elements per filter

            float val =
                __ldg(&input[input_offset + 0])         * __ldg(&weight[weight_base + 0]) +
                __ldg(&input[input_offset + 1])         * __ldg(&weight[weight_base + 1]) +
                __ldg(&input[input_offset + 2])         * __ldg(&weight[weight_base + 2]) +
                __ldg(&input[input_offset + width + 0]) * __ldg(&weight[weight_base + 3]) +
                __ldg(&input[input_offset + width + 1]) * __ldg(&weight[weight_base + 4]) +
                __ldg(&input[input_offset + width + 2]) * __ldg(&weight[weight_base + 5]) +
                __ldg(&input[input_offset + 2*width + 0]) * __ldg(&weight[weight_base + 6]) +
                __ldg(&input[input_offset + 2*width + 1]) * __ldg(&weight[weight_base + 7]) +
                __ldg(&input[input_offset + 2*width + 2]) * __ldg(&weight[weight_base + 8]);
            sum += val;
        }
    } else {
        // Generic convolution: calculate effective convolution window boundaries
        int h_start = max(0, -h * stride + padding);
        int h_end = min(kernel_h, height - h * stride + padding);
        int w_start = max(0, -w * stride + padding);
        int w_end = min(kernel_w, width - w * stride + padding);
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = h_start; kh < h_end; kh++) {
                int h_in = h * stride + kh - padding;
                #pragma unroll
                for (int kw = w_start; kw < w_end; kw++) {
                    int w_in = w * stride + kw - padding;
                    float in_val = __ldg(&input[((b * in_channels + ic) * height + h_in) * width + w_in]);
                    float wt_val = __ldg(&weight[((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw]);
                    sum += in_val * wt_val;
                }
            }
        }
    }

    output[((b * out_channels + oc) * out_h + h) * out_w + w] = sum;
}


torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");

    if(dilation != 1 || groups != 1) {
        return torch::conv2d(x, weight, bias, {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int out_h = (height + 2 * padding - kernel_h) / stride + 1;
    int out_w = (width + 2 * padding - kernel_w) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());

    constexpr int BLOCK_SIZE = 256;
    int total = batch_size * out_channels * out_h * out_w;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    conv2d_kernel<<<blocks, BLOCK_SIZE>>>(
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

    if(bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution");
}
