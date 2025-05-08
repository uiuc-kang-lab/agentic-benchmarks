#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Block dimensions for spatial tiling
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8

// This kernel manually unrolls the loops over kernel height and width
// to reduce loop overhead for critical inner loops.
__global__ void depthwise_conv2d_kernel_unroll(
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
    int x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    int temp = blockIdx.z;
    int c_out = temp % out_channels;
    int b = temp / out_channels;

    if (x >= out_w || y >= out_h || b >= batch_size) return;

    int g = c_out / channels_per_group;
    int m = c_out % channels_per_group;

    float sum = 0.0f;
    
    // Manually unroll the loops for kernel height and width
    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        int h_in = y * stride_h - padding_h + kh * dilation_h;
        #pragma unroll
        for (int kw = 0; kw < kernel_w; ++kw) {
            int w_in = x * stride_w - padding_w + kw * dilation_w;
            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                int in_index = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
                int weight_index = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
                sum += input[in_index] * weight[weight_index];
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    int out_index = ((b * out_channels + c_out) * out_h + y) * out_w + x;
    output[out_index] = sum;
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

    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 blocks(
        (out_w + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (out_h + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        batch_size * out_channels
    );

    depthwise_conv2d_kernel_unroll<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Depthwise Conv2D forward with loop unrolling (CUDA)");
}
