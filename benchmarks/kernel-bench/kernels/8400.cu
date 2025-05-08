#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Optimized custom CUDA kernel for ConvTranspose2d when groups==1 and dilation==1
// Uses a grid-stride loop for efficiency and minimal branching

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_height,
    const int kernel_width,
    const int output_height,
    const int output_width,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int total_elements = batch_size * out_channels * output_height * output_width;

    for (int idx = tid; idx < total_elements; idx += total_threads) {
        // Compute output coordinates
        int w = idx % output_width;
        int h = (idx / output_width) % output_height;
        int c = (idx / (output_width * output_height)) % out_channels;
        int b = idx / (output_width * output_height * out_channels);

        scalar_t sum = 0;
        int batch_offset = b * in_channels * input_height * input_width;

        // Loop over input channels and kernel elements
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            int in_ch_offset = in_ch * input_height * input_width;
            int weight_in_offset = in_ch * out_channels * kernel_height * kernel_width;
            
            // Unroll over kernel height
            #pragma unroll
            for (int kh = 0; kh < kernel_height; kh++) {
                // Calculate corresponding input y coordinate
                int in_y = h + pad_h - kh;
                if (in_y % stride_h != 0) continue;
                int input_y = in_y / stride_h;
                if (input_y < 0 || input_y >= input_height) continue;

                // Loop over kernel width
                for (int kw = 0; kw < kernel_width; kw++) {
                    int in_x = w + pad_w - kw;
                    if (in_x % stride_w != 0) continue;
                    int input_x = in_x / stride_w;
                    if (input_x < 0 || input_x >= input_width) continue;

                    scalar_t input_val = input[batch_offset + in_ch_offset + input_y * input_width + input_x];
                    scalar_t weight_val = weight[weight_in_offset + c * kernel_height * kernel_width + kh * kernel_width + kw];
                    sum += input_val * weight_val;
                }
            }
        }
        
        // Add bias if provided
        if (bias != nullptr)
            sum += bias[c];
        
        output[idx] = sum;
    }
}


// The main API called from Python
// This integrated implementation uses the custom kernel when possible,
// and falls back to PyTorch's built-in conv_transpose2d when parameters do not meet
// the assumptions (e.g., groups != 1 or dilation != 1).

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    // Fallback to the built-in implementation for unsupported parameters
    if (groups != 1 || dilation[0] != 1 || dilation[1] != 1) {
        return at::conv_transpose2d(x,
                                    weight,
                                    bias.value_or(torch::Tensor()),
                                    stride,
                                    padding,
                                    output_padding,
                                    groups,
                                    dilation);
    }

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_height = x.size(2);
    const int input_width = x.size(3);
    const int out_channels = weight.size(1);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);

    // Compute output dimensions based on the transpose convolution formula
    const int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0];
    const int output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, x.options());

    const int block_size = 256;
    const int total_elements = batch_size * out_channels * output_height * output_width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    // Cap grid size to ensure compatibility
    const int grid_size = std::min(num_blocks, 65535);

    // Get bias pointer if bias is defined
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    // Dispatch over float/double types
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<grid_size, block_size>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias_ptr,
            output.data_ptr<scalar_t>(),
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
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "Integrated Efficient ConvTranspose2D forward (CUDA)");
}
