#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel implements a gather-based transposed 2D convolution where each thread computes one output element.
// By gathering contributions from input elements, we avoid the need for expensive atomicAdd operations used in scatter approaches.
// It also fuses bias addition directly in the kernel and uses a grid-stride loop for better load balancing.

template <typename scalar_t>
__global__ void conv_transpose2d_gather_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias, // bias can be nullptr
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
    const int output_padding,
    const int groups,
    const int dilation,
    const int out_height,
    const int out_width
) {
    // Total number of output elements
    const int total = batch_size * out_channels * out_height * out_width;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        // Unravel the flat index to (b, oc, oh, ow)
        int n = idx;
        const int ow = n % out_width; n /= out_width;
        const int oh = n % out_height; n /= out_height;
        const int oc = n % out_channels; n /= out_channels;
        const int b = n;

        // Determine group indexes
        const int out_channels_per_group = out_channels / groups;
        const int g = oc / out_channels_per_group;
        const int oc_group = oc % out_channels_per_group;

        const int in_channels_per_group = in_channels / groups;
        const int ic_start = g * in_channels_per_group;

        // Start with bias if provided
        scalar_t sum = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);

        // Loop over the kernel spatial dimensions
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // The corresponding input coordinates are computed by inverting the convolution formula
                // for transposed convolution: oh = h_in * stride - padding + kh * dilation
                // Rearranged: h_in = (oh + padding - kh*dilation) / stride, and similarly for width.
                int h_in_calc = oh + padding - kh * dilation;
                int w_in_calc = ow + padding - kw * dilation;
                
                // Check that the indices align with stride steps
                if ((h_in_calc % stride) != 0 || (w_in_calc % stride) != 0) continue;
                int h_in = h_in_calc / stride;
                int w_in = w_in_calc / stride;
                
                // Validate input coordinate bounds
                if (h_in < 0 || h_in >= in_height || w_in < 0 || w_in >= in_width) continue;

                // Accumulate over the appropriate input channels in the group
                for (int ic = 0; ic < in_channels_per_group; ++ic) {
                    int cur_ic = ic_start + ic;

                    int input_idx = b * (in_channels * in_height * in_width) + 
                                    cur_ic * (in_height * in_width) + 
                                    h_in * in_width + w_in;
                    scalar_t input_val = input[input_idx];

                    // Weight indexing: Weight shape is [in_channels, out_channels/groups, kernel_h, kernel_w]
                    int weight_idx = cur_ic * (out_channels_per_group * kernel_h * kernel_w) +
                                     oc_group * (kernel_h * kernel_w) +
                                     kh * kernel_w + kw;
                    scalar_t weight_val = weight[weight_idx];

                    sum += input_val * weight_val;
                }
            }
        }

        // Write the computed output element
        output[idx] = sum;
    }
}


// Forward function to prepare output tensor and launch the kernel

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation = 1
) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);

    // Weight shape: [in_channels, out_channels/groups, kernel_h, kernel_w]
    const int out_channels = weight.size(1) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    // Compute output dimensions following standard transposed convolution computation
    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    const int total_elements = output.numel();
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_gather_cuda", ([&] {
        conv_transpose2d_gather_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            (bias.has_value() && bias->defined()) ? bias->data_ptr<scalar_t>() : nullptr,
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
    m.def("forward", &forward, "Optimized Transposed 2D Convolution using Gather Approach (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}
