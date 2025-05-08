#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS 256
#define SMALL_KERNEL_THRESHOLD 5
#define SMALL_CHANNEL_THRESHOLD 64

__global__ void warp_optimized_conv_transpose2d_kernel(
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
    int pad_w)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_elements = batch_size * out_channels * output_height * output_width;

    for (int idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        int w = idx % output_width;
        int h = (idx / output_width) % output_height;
        int c = (idx / (output_width * output_height)) % out_channels;
        int b = idx / (output_width * output_height * out_channels);

        float sum = 0.0f;
        int batch_offset = b * in_channels * input_height * input_width;
        int out_ch_offset = c * kernel_height * kernel_width;

        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            int in_ch_offset = in_ch * input_height * input_width;
            int weight_in_offset = in_ch * out_channels * kernel_height * kernel_width;

            for (int kh = 0; kh < kernel_height; kh++) {
                int in_y = h + pad_h - kh;
                bool y_valid = (in_y % stride_h == 0) && (in_y / stride_h) < input_height && (in_y / stride_h) >= 0;
                if (y_valid) {
                    int input_h = in_y / stride_h;

                    for (int kw = 0; kw < kernel_width; kw++) {
                        int in_x = w + pad_w - kw;
                        bool x_valid = (in_x % stride_w == 0) && (in_x / stride_w) < input_width && (in_x / stride_w) >= 0;
                        if (x_valid) {
                            int input_w = in_x / stride_w;

                            float input_val = input[batch_offset + in_ch_offset + input_h * input_width + input_w];
                            float weight_val = weight[weight_in_offset + out_ch_offset + kh * kernel_width + kw];
                            sum += input_val * weight_val;
                        }
                    }
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[c];
        }

        output[idx] = sum;
    }
}


torch::Tensor warp_optimized_conv_transpose2d_cuda(
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

    const int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + 
                              kernel_height + output_padding[0];
    const int output_width = (input_width - 1) * stride[1] - 2 * padding[1] + 
                             kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, x.options());

    int total_sum_elems = in_channels * kernel_height * kernel_width;
    int blocks_per_out = (total_sum_elems + THREADS - 1) / THREADS;
    int num_output_pixels = batch_size * out_channels * output_height * output_width;
    int total_blocks = num_output_pixels * blocks_per_out;

    dim3 blocks(total_blocks);
    dim3 threads(THREADS);

    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    warp_optimized_conv_transpose2d_kernel<< <blocks, threads >> >(
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
    m.def("forward", &warp_optimized_conv_transpose2d_cuda, "Warp Optimized ConvTranspose2D forward (CUDA)");
}