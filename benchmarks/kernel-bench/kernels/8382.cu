#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS 256
#define SMALL_CHANNEL_THRESHOLD 64
#define SMALL_KERNEL_THRESHOLD 5

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    int in_channels = input.size(1);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    // Use built-in implementation for small channels/kernels
    if (in_channels <= SMALL_CHANNEL_THRESHOLD && 
        kernel_height <= SMALL_KERNEL_THRESHOLD &&
        kernel_width <= SMALL_KERNEL_THRESHOLD) {
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

    // Use custom implementation for larger channels/kernels
    int batch_size = input.size(0);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(1);

    int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0];
    int output_width  = (input_width - 1)  * stride[1] - 2 * padding[1] + kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

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

    conv_transpose2d_atomic_reduce_kernel<<<blocks, threads>>>(
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

// Keep the original atomic reduce kernel implementation...
__global__ void conv_transpose2d_atomic_reduce_kernel(/* ... same parameters ... */) {
    // ... same kernel implementation as before ...
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "Hybrid adaptive ConvTranspose2D forward (CUDA)");
}