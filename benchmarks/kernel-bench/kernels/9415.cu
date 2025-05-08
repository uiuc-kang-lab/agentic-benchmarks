#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Grid-stride loop based 2D convolution kernel
__global__ void grid_stride_conv2d_kernel(
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
    int dilation_w) 
{
    // Total number of output elements
    int total_elements = batch_size * out_channels * height_out * width_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    // Each thread handles multiple output elements using a grid-stride loop
    for (int i = idx; i < total_elements; i += grid_stride) {
        int index = i;
        // Derive output coordinates: b, oc, h, w from the flattened index
        int w = index % width_out;
        index /= width_out;
        int h = index % height_out;
        index /= height_out;
        int oc = index % out_channels;
        int b = index / out_channels;

        float sum = (bias_ptr != nullptr) ? bias_ptr[oc] : 0.0f;

        // Compute the origin position in input with padding and stride
        int h_in_origin = h * stride - pad_h;
        int w_in_origin = w * stride - pad_w;

        // Loop over input channels and kernel window
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                int h_in = h_in_origin + kh * dilation_h;
                if (h_in < 0 || h_in >= input_height)
                    continue;
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int w_in = w_in_origin + kw * dilation_w;
                    if (w_in < 0 || w_in >= input_width)
                        continue;

                    int x_index = b * in_channels * input_height * input_width +
                                  ic * input_height * input_width +
                                  h_in * input_width + w_in;
                    int w_index = oc * in_channels * kernel_h * kernel_w +
                                  ic * kernel_h * kernel_w +
                                  kh * kernel_w + kw;
                    sum += x[x_index] * weight[w_index];
                }
            }
        }

        // Write the computed value to the output tensor
        output[i] = sum;
    }
}

// Forward function exposed to PyTorch
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
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

    int total_elements = batch_size * out_channels * height_out * width_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    grid_stride_conv2d_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Conv2D forward (CUDA)");
}
