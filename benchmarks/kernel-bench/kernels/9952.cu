#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// Kernel that assigns one output pixel per warp. Each warp's threads compute partial sums
// over the convolution kernel elements and then use warp-level primitives for reduction.
__global__ void depthwise_conv2d_kernel_warp_reduce(
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
    // Compute global thread id
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // Each warp processes one output pixel
    int warp_id = thread_id / WARP_SIZE;
    int lane = thread_id % WARP_SIZE;

    int total_outputs = batch_size * out_channels * out_h * out_w;
    if (warp_id >= total_outputs) return;

    // Decode warp_id into output tensor coordinates: b, c_out, h_out, w_out
    int tmp = warp_id;
    int w_out = tmp % out_w;
    tmp /= out_w;
    int h_out = tmp % out_h;
    tmp /= out_h;
    int c_out = tmp % out_channels;
    int b = tmp / out_channels;

    int g = c_out / channels_per_group;  // group index

    int kernel_size = kernel_h * kernel_w;
    float sum = 0.0f;

    // Each thread in the warp handles a subset of kernel elements
    for (int k = lane; k < kernel_size; k += WARP_SIZE) {
        int kh = k / kernel_w;
        int kw = k % kernel_w;
        int h_in = h_out * stride_h - padding_h + kh * dilation_h;
        int w_in = w_out * stride_w - padding_w + kw * dilation_w;
        if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
            int input_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
            int weight_idx = ((g * channels_per_group + (c_out % channels_per_group)) * kernel_h + kh) * kernel_w + kw;
            sum += input[input_idx] * weight[weight_idx];
        }
    }

    // Use warp-level reduction entirely with __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Lane 0 writes the final sum
    if (lane == 0) {
        if (bias != nullptr) {
            sum += bias[c_out];
        }
        output[warp_id] = sum;
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
    if (bias.has_value()) {
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

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias->data_ptr<float>();
    }

    int total_outputs = batch_size * out_channels * out_h * out_w;  // one per warp
    int total_threads = total_outputs * WARP_SIZE;  // each warp has WARP_SIZE threads
    int threads = 256; // must be a multiple of WARP_SIZE
    int blocks = (total_threads + threads - 1) / threads;

    depthwise_conv2d_kernel_warp_reduce<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
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
    m.def("forward", &forward, "Depthwise Conv2D forward with warp-level reduction (CUDA)");
}
