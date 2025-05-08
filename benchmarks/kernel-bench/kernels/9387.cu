#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Modular device function to compute convolution sum for a single output element
__device__ inline float compute_conv2d_sum(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    int b,
    int oc,
    int h_out,
    int w_out,
    int in_channels,
    int input_height,
    int input_width,
    int kernel_h,
    int kernel_w,
    int stride,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w) {

    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride + kh * dilation_h - pad_h;
                int w_in = w_out * stride + kw * dilation_w - pad_w;
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    int x_index = b * (in_channels * input_height * input_width)
                                + ic * (input_height * input_width)
                                + h_in * input_width + w_in;
                    int weight_index = oc * (in_channels * kernel_h * kernel_w)
                                     + ic * (kernel_h * kernel_w)
                                     + kh * kernel_w + kw;
                    sum += x[x_index] * weight[weight_index];
                }
            }
        }
    }
    return sum;
}

// Kernel that leverages the modular device function for computing convolution
__global__ void modular_conv2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int height_out,
    int width_out,
    int stride,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w) {

    // Map threads to output width, height, and output channel
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;

    if (w_out >= width_out || h_out >= height_out || oc >= out_channels) return;

    // Loop over batch dimension
    for (int b = 0; b < batch_size; ++b) {
        // Initialize with bias if provided
        float value = (bias != nullptr) ? bias[oc] : 0.0f;
        // Add convolution sum computed by the modular device function
        value += compute_conv2d_sum(
            x, weight, b, oc, h_out, w_out,
            in_channels, input_height, input_width,
            kernel_h, kernel_w, stride, pad_h, pad_w, dilation_h, dilation_w);
        
        int out_index = b * (out_channels * height_out * width_out)
                      + oc * (height_out * width_out)
                      + h_out * width_out + w_out;
        output[out_index] = value;
    }
}

// Forward function wrapping the CUDA kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,  // optional bias
    int stride,
    std::tuple<int, int> padding,
    std::tuple<int, int> dilation) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        bias_ptr = bias->data_ptr<float>();
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);

    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int pad_h = std::get<0>(padding);
    int pad_w = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    int height_out = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int width_out = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());

    // Configure grid and block dimensions
    dim3 threads(16, 16);
    dim3 blocks((width_out + threads.x - 1) / threads.x,
                (height_out + threads.y - 1) / threads.y,
                out_channels);

    modular_conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_height,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        height_out,
        width_out,
        stride,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular Conv2D forward (CUDA) with device function refactor");
}
