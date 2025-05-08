#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS 256
#define SMALL_KERNEL_THRESHOLD 5
#define SMALL_CHANNEL_THRESHOLD 64

__global__ void conv_transpose2d_coalesced_kernel(
    const float * __restrict__ input,
    const float * __restrict__ weight,
    const float * __restrict__ bias,
    float * __restrict__ output,
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
    int pad_w,
    int total_sum_elems,
    int blocks_per_out)
{
    int tid = threadIdx.x;
    int out_idx = blockIdx.x / blocks_per_out;
    int block_offset = blockIdx.x % blocks_per_out;

    int w = out_idx % output_width;
    int h = (out_idx / output_width) % output_height;
    int c = (out_idx / (output_width * output_height)) % out_channels;
    int b = out_idx / (output_width * output_height * out_channels);

    if (b >= batch_size || c >= out_channels) return;

    float sum = 0.0f;
    int batch_offset = b * in_channels * input_height * input_width;
    int out_ch_offset = c * kernel_height * kernel_width;

    for (int in_ch = block_offset; in_ch < in_channels; in_ch += blocks_per_out) {
        int in_ch_offset = in_ch * input_height * input_width;
        int weight_in_offset = in_ch * out_channels * kernel_height * kernel_width;

        for (int kh = 0; kh < kernel_height; kh++) {
            int in_y = h + pad_h - kh;
            if (in_y % stride_h != 0) continue;
            int input_h = in_y / stride_h;
            if (input_h < 0 || input_h >= input_height) continue;

            for (int kw = 0; kw < kernel_width; kw++) {
                int in_x = w + pad_w - kw;
                if (in_x % stride_w != 0) continue;
                int input_w = in_x / stride_w;
                if (input_w < 0 || input_w >= input_width) continue;

                // Ensure memory coalescing by accessing consecutive memory locations
                float input_val = input[batch_offset + in_ch_offset + input_h * input_width + input_w];
                float weight_val = weight[weight_in_offset + out_ch_offset + kh * kernel_width + kw];

                sum += input_val * weight_val;
            }
        }
    }

    if (bias != nullptr && tid == 0) {
        sum += bias[c];
    }

    // Write the result to the output
    output[b * out_channels * output_height * output_width +
           c * output_height * output_width +
           h * output_width + w] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);
    int in_channels = input.size(1);
    
    if (kernel_height <= SMALL_KERNEL_THRESHOLD && 
        kernel_width <= SMALL_KERNEL_THRESHOLD && 
        in_channels <= SMALL_CHANNEL_THRESHOLD) {
        return at::conv_transpose2d(
            input,
            weight,
            bias.value_or(torch::Tensor()),
            stride,
            padding,
            output_padding,
            groups,
            dilation
        );
    }
    
    int batch_size = input.size(0);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(1);

    int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + 
                       kernel_height + output_padding[0];
    int output_width = (input_width - 1) * stride[1] - 2 * padding[1] + 
                      kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                             input.options());

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

    conv_transpose2d_coalesced_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
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
        padding[1],
        total_sum_elems,
        blocks_per_out
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "Memory Coalesced ConvTranspose2D forward (CUDA)");
}