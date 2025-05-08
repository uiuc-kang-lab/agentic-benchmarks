#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes each output element by gathering contributions from input and weight.
// It combines the grid-stride loop and early exit conditionals from the two kernels, and avoids atomic operations.

template <typename scalar_t>
__global__ void conv_transpose2d_gather_unified_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,  // can be nullptr if no bias is provided
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding,
    const int output_padding,  // used only for output size computation
    const int groups,
    const int dilation,
    const int out_height,
    const int out_width
) {
    // Total number of output elements
    int total_outputs = batch_size * out_channels * out_height * out_width;
    int gridStride = blockDim.x * gridDim.x;

    // Precomputed sizes
    const int in_image_size = in_height * in_width;
    const int out_image_size = out_height * out_width;
    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_outputs; idx += gridStride) {
        // Unravel the flat index to 4D indices: b, oc, oh, ow
        int tmp = idx;
        int ow = tmp % out_width;
        tmp /= out_width;
        int oh = tmp % out_height;
        tmp /= out_height;
        int oc = tmp % out_channels;
        int b = tmp / out_channels;

        // Determine group information
        int g = oc / out_channels_per_group;
        int oc_group = oc % out_channels_per_group;

        // Initialize with bias if provided
        scalar_t sum = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);

        // Loop over the kernel height and width dimensions
        for (int kh = 0; kh < kernel_h; ++kh) {
            // Compute the corresponding input row index condition:
            //   oh = h_in * stride - padding + kh * dilation   =>   h_in = (oh + padding - kh*dilation) / stride
            int h_val = oh + padding - kh * dilation;
            if (h_val < 0 || h_val >= stride * in_height) continue;  // out-of-bound
            if (h_val % stride != 0) continue;  // misalignment
            int h_in = h_val / stride;
            if (h_in < 0 || h_in >= in_height) continue;

            for (int kw = 0; kw < kernel_w; ++kw) {
                int w_val = ow + padding - kw * dilation;
                if (w_val < 0 || w_val >= stride * in_width) continue;
                if (w_val % stride != 0) continue;
                int w_in = w_val / stride;
                if (w_in < 0 || w_in >= in_width) continue;

                // Accumulate contributions across the input channels in the current group
                int ic_start = g * in_channels_per_group;
                for (int ic = 0; ic < in_channels_per_group; ++ic) {
                    int cur_ic = ic_start + ic;

                    // Compute indices for input and weight
                    int input_idx = b * (in_channels * in_image_size) + cur_ic * in_image_size + h_in * in_width + w_in;
                    int weight_idx = cur_ic * (out_channels_per_group * kernel_h * kernel_w)
                                   + oc_group * (kernel_h * kernel_w)
                                   + kh * kernel_w + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }

        // Write the result into the output tensor
        int output_idx = b * (out_channels * out_image_size) + oc * out_image_size + oh * out_width + ow;
        output[output_idx] = sum;
    }
}

// Forward function: precomputes output dimensions and launches the unified gather kernel

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation = 1
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);

    // Weight is assumed to have shape: [in_channels, out_channels/groups, kernel_h, kernel_w]
    const int out_channels = weight.size(1) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    // Compute output dimensions according to transposed convolution formula
    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    // Determine bias pointer (if provided)
    const torch::Tensor bias_ptr = (bias.has_value() && bias->defined()) ? bias.value() : torch::Tensor();
    
    const int THREADS = 256;
    const int total_outputs = batch_size * out_channels * out_height * out_width;
    const int BLOCKS = (total_outputs + THREADS - 1) / THREADS;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_gather_unified_cuda", ([&] {
        conv_transpose2d_gather_unified_kernel<scalar_t><<<BLOCKS, THREADS>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias_ptr.defined() ? bias_ptr.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
            out_height,
            out_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Unified Gather Transposed 2D Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}
