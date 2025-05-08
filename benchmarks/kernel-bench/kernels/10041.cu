#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using warp-level reduction with __shfl_down_sync to reduce the inner convolution sum
__global__ void depthwise_conv2d_kernel_warp(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int channels_per_group
) {
    // Each warp computes one output element
    // Compute global warp ID
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;  // lane id within the warp

    int total_output = batch_size * out_channels * out_h * out_w;
    if (warpId >= total_output) return;

    // Decode warpId into output indices
    int tmp = warpId;
    int w_out = tmp % out_w;
    tmp /= out_w;
    int h_out = tmp % out_h;
    tmp /= out_h;
    int c_out = tmp % out_channels;
    int b = tmp / out_channels;

    // Determine corresponding group and channel within the group
    int g = c_out / channels_per_group;
    int m = c_out % channels_per_group;

    int kernel_size = kernel_h * kernel_w;
    float sum = 0.0f;
    // Each warp lane accumulates partial sum from a subset of the kernel window
    for (int idx = lane; idx < kernel_size; idx += 32) {
        int kh = idx / kernel_w;
        int kw = idx % kernel_w;
        int h_in = h_out * stride_h - padding_h + kh * dilation_h;
        int w_in = w_out * stride_w - padding_w + kw * dilation_w;
        if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
            int input_index = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
            int weight_index = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
            sum += __ldg(&input[input_index]) * __ldg(&weight[weight_index]);
        }
    }
    
    // Warp-level reduction using __shfl_down_sync
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Lane 0 writes the final result
    if (lane == 0) {
        if (bias != nullptr) {
            sum += bias[c_out];
        }
        output[warpId] = sum;
    }
}


torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    if(bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = groups * weight.size(1);
    int channels_per_group = out_channels / groups;

    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    int total_output = batch_size * out_channels * out_h * out_w; // one warp per output element
    // Choose 256 threads per block (8 warps per block)
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int blocks = (total_output + warps_per_block - 1) / warps_per_block;

    depthwise_conv2d_kernel_warp<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp Reduced Depthwise Conv2d forward (CUDA)");
}
