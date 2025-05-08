#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses a flattened 1D grid-stride loop to map each thread to an output element.
// This simplifies thread and block indexing by avoiding 3D grid calculations and reduces overhead.
__global__ void depthwise_conv2d_flat_index_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int channels_per_group
) {
    const int total = batch_size * out_channels * out_h * out_w;
    // Each thread processes multiple output elements via grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int tmp = idx;
        int w_out = tmp % out_w; 
        tmp /= out_w;
        int h_out = tmp % out_h; 
        tmp /= out_h;
        int c_out = tmp % out_channels;
        int b = tmp / out_channels;

        int g = c_out / channels_per_group;
        int m = c_out % channels_per_group;

        float sum = 0.0f;
        // Iterate over the kernel window
        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_in = h_out * stride_h - padding_h + kh * dilation_h;
            if (h_in >= 0 && h_in < in_h) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int w_in = w_out * stride_w - padding_w + kw * dilation_w;
                    if (w_in >= 0 && w_in < in_w) {
                        int input_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
                        int weight_idx = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        if (bias) {
            sum += bias[c_out];
        }
        output[idx] = sum;
    }
}

// Forward function that sets up kernel launch parameters based on the total output size.
// It computes a 1D grid of threads, so that each thread computes one or multiple output elements via a grid-stride loop.

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

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);

    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels = groups * weight.size(1);
    const int channels_per_group = out_channels / groups;

    const int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    int total = batch_size * out_channels * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_flat_index_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Depthwise Conv2D with flat index (CUDA)");
}
