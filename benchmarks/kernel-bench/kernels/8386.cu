#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized kernel with manual loop unrolling for critical loops
__global__ void conv_transpose2d_kernel_unrolled(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int output_height,
    int output_width,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w) {

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int linear_idx = blockIdx.z;
    int batch = linear_idx / out_channels;
    int out_ch = linear_idx % out_channels;

    if (out_x < output_width && out_y < output_height && batch < batch_size) {
        float sum = 0.0f;
        // Unroll loop over input channels, kernel spatial dimensions
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            #pragma unroll 4
            for (int kh = 0; kh < kernel_height; kh++) {
                #pragma unroll 4
                for (int kw = 0; kw < kernel_width; kw++) {
                    int in_x = out_x + pad_w - kw;
                    int in_y = out_y + pad_h - kh;
                    if (in_x % stride_w == 0 && in_y % stride_h == 0) {
                        in_x /= stride_w;
                        in_y /= stride_h;
                        if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
                            float input_val = input[batch * in_channels * input_height * input_width +
                                                      in_ch * input_height * input_width +
                                                      in_y * input_width + in_x];

                            float weight_val = weight[in_ch * out_channels * kernel_height * kernel_width +
                                                        out_ch * kernel_height * kernel_width +
                                                        kh * kernel_width + kw];

                            sum += input_val * weight_val;
                        }
                    }
                }
            }
        }
        if (bias) {
            sum += bias[out_ch];
        }

        output[batch * out_channels * output_height * output_width +
               out_ch * output_height * output_width +
               out_y * output_width + out_x] = sum;
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_height = x.size(2);
    const int input_width = x.size(3);
    const int out_channels = weight.size(1);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);

    const int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0];
    const int output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, x.options());

    dim3 threads(16, 16, 1);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * out_channels
    );

    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    conv_transpose2d_kernel_unrolled<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        output_height,
        output_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1]
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "Optimized ConvTranspose2D forward with unrolled loops (CUDA)");
}
