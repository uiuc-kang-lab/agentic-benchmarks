#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel that uses warp-level reduction to sum contributions for each output pixel
// Each warp (32 threads) collaborates to compute one output element

__global__ void conv_transpose2d_kernel_warp(
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
    int pad_w,
    int total_reduce_elements) {

    // Global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Each warp (32 threads) handles one output element
    int warp_id = tid / 32;
    int lane = tid % 32;

    // Total number of output elements
    int total_outputs = batch_size * out_channels * output_height * output_width;
    if (warp_id >= total_outputs)
        return;

    // Decode warp_id into (batch, out_channel, out_y, out_x)
    int output_size = output_height * output_width;
    int oc_size = out_channels * output_size;
    int b = warp_id / oc_size; 
    int rem = warp_id % oc_size;
    int oc = rem / output_size;
    rem = rem % output_size;
    int out_y = rem / output_width;
    int out_x = rem % output_width;

    float sum = 0.0f;
    // Loop over the reduction domain: for each in_channel and kernel spatial position
    // total_reduce_elements = in_channels * kernel_height * kernel_width
    for (int idx = lane; idx < total_reduce_elements; idx += 32) {
        int in_ch = idx / (kernel_height * kernel_width);
        int r = idx % (kernel_height * kernel_width);
        int kh = r / kernel_width;
        int kw = r % kernel_width;

        // Compute the corresponding input coordinates
        int in_x = out_x + pad_w - kw;
        int in_y = out_y + pad_h - kh;

        // Ensure alignment with the stride
        if ((in_x % stride_w) == 0 && (in_y % stride_h) == 0) {
            in_x /= stride_w;
            in_y /= stride_h;
            if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
                int input_offset = b * in_channels * input_height * input_width +
                                   in_ch * input_height * input_width +
                                   in_y * input_width + in_x;
                int weight_offset = in_ch * out_channels * kernel_height * kernel_width +
                                    oc * kernel_height * kernel_width +
                                    kh * kernel_width + kw;
                float val = input[input_offset] * weight[weight_offset];
                sum += val;
            }
        }
    }

    // Warp-level reduction using __shfl_down_sync
    // Use full warp mask 0xffffffff
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Lane 0 writes the result
    if (lane == 0) {
        if (bias != nullptr) {
            sum += bias[oc];
        }
        int output_offset = b * out_channels * output_height * output_width +
                            oc * output_height * output_width +
                            out_y * output_width + out_x;
        output[output_offset] = sum;
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

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);
    int out_channels = weight.size(1);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0];
    int output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, x.options());

    int total_output = batch_size * out_channels * output_height * output_width;
    int total_reduce_elements = in_channels * kernel_height * kernel_width;

    // Each warp (32 threads) computes one output element
    int total_threads = total_output * 32;
    int blockSize = 256;
    int gridSize = (total_threads + blockSize - 1) / blockSize;

    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    conv_transpose2d_kernel_warp<<<gridSize, blockSize>>>(
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
        padding[1],
        total_reduce_elements);

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward with warp-level reduction (CUDA)");
}
