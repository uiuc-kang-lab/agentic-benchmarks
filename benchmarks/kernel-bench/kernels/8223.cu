#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel to initialize the output tensor with bias (if provided) or zeros
// Output tensor shape: [batch_size, out_channels, out_height, out_width]
template <typename scalar_t>
__global__ void init_output_kernel(
    scalar_t *output,
    const int total,
    const scalar_t *bias,
    const int out_channels,
    const int out_height,
    const int out_width
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total) {
        if (bias != nullptr) {
            int tmp = index;
            int ow = tmp % out_width;
            tmp /= out_width;
            int oh = tmp % out_height;
            tmp /= out_height;
            int oc = tmp % out_channels;
            // int b = tmp / out_channels; // not used
            output[index] = bias[oc];
        } else {
            output[index] = static_cast<scalar_t>(0);
        }
    }
}

// Scatter-based transposed convolution kernel using atomicAdd
// Each thread processes one input element and scatters its contributions
// to the corresponding output locations using atomic operations.
// Weight shape is assumed to be: [in_channels, out_channels_per_group, kernel_h, kernel_w]
// and output channels = out_channels_per_group * groups.

template <typename scalar_t>
__global__ void conv_transpose2d_scatter_kernel(
    const scalar_t * __restrict__ input,
    const scalar_t * __restrict__ weight,
    scalar_t * __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding,
    const int groups,
    const int dilation,
    const int out_height,
    const int out_width
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_input = batch_size * in_channels * in_height * in_width;
    if (index >= total_input) return;

    // Unravel index for input: index -> b, ic, h, w
    int tmp = index;
    int w = tmp % in_width;
    tmp /= in_width;
    int h = tmp % in_height;
    tmp /= in_height;
    int ic = tmp % in_channels;
    int b = tmp / in_channels;
    int b_offset = b * (out_channels * out_height * out_width);
    int ic_base = ic * (out_channels_per_group * kernel_h * kernel_w);

    // Determine group and corresponding out_channels per group
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    int g = ic / in_channels_per_group;  

    scalar_t input_val = input[index];

    // Iterate over kernel spatial dimensions
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            int oh = h * stride - padding + kh * dilation;
            int ow = w * stride - padding + kw * dilation;
            if (oh < 0 || oh >= out_height || ow < 0 || ow >= out_width)
                continue;
            
            // For each output channel in the group
            for (int oc_offset = 0; oc_offset < out_channels_per_group; oc_offset++) {
                // Weight index: weight has shape [in_channels, out_channels_per_group, kernel_h, kernel_w]
                int weight_index = ic * (out_channels_per_group * kernel_h * kernel_w)
                                   + oc_offset * (kernel_h * kernel_w)
                                   + kh * kernel_w + kw;
                scalar_t prod = input_val * weight[weight_index];
                
                int oc = g * out_channels_per_group + oc_offset;
                int out_index = b * (out_channels * out_height * out_width)
                                + oc * (out_height * out_width)
                                + oh * out_width + ow;
                atomicAdd(&output[out_index], prod);
            }
        }
    }
}

// Forward function: Initializes output and then launches the scatter kernel
// to perform the transposed convolution operation.

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
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);

    // Weight shape: [in_channels, out_channels_per_group, kernel_h, kernel_w]
    const int out_channels = weight.size(1) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    // Compute output spatial dimensions for transposed convolution
    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, x.options());

    // Initialize output with bias if provided, or zeros otherwise
    int total_output = output.numel();
    const int threads = 256;
    const int init_blocks = (total_output + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "init_output_kernel", ([&] {
        init_output_kernel<scalar_t><<<init_blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            total_output,
            (bias.has_value() && bias->defined()) ? bias->data_ptr<scalar_t>() : nullptr,
            out_channels,
            out_height,
            out_width
        );
    }));

    // Launch scatter kernel over input elements
    int total_input = x.numel();
    const int scatter_blocks = (total_input + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_scatter_cuda", ([&] {
        conv_transpose2d_scatter_kernel<scalar_t><<<scatter_blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
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
            groups,
            dilation,
            out_height,
            out_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 2D convolution scatter (CUDA) with minimal atomic usage",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}
