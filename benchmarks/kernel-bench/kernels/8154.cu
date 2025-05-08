#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Scatter-based transposed convolution kernel.
// Each thread processes one input element and scatters its contributions to multiple output locations.
// Atomic operations (atomicAdd) are used to avoid race conditions when multiple threads update the same output element.

template <typename scalar_t>
__global__ void conv_transpose2d_scatter_atomic_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
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
    const int output_padding,  // used to compute output dimensions
    const int groups,
    const int dilation,
    const int out_height,
    const int out_width
) {
    // Total number of input elements
    int total_input = batch_size * in_channels * in_height * in_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;

    for (; idx < total_input; idx += gridStride) {
        // Map flat index to (b, ic, h, w) in input
        int temp = idx;
        int w_in = temp % in_width;
        temp /= in_width;
        int h_in = temp % in_height;
        temp /= in_height;
        int ic = temp % in_channels;
        temp /= in_channels;
        int b = temp;

        // Determine group indices
        int in_channels_per_group = in_channels / groups; 
        int out_channels_per_group = out_channels / groups; 
        int g = ic / in_channels_per_group; 

        // Load the input value
        scalar_t in_val = input[idx]; // input[b, ic, h_in, w_in]

        // For each kernel element
        for (int kh = 0; kh < kernel_h; ++kh) {
            // Compute the output row index
            int oh = h_in * stride - padding + kh * dilation;
            if (oh < 0 || oh >= out_height) continue;
            
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Compute the output column index
                int ow = w_in * stride - padding + kw * dilation;
                if (ow < 0 || ow >= out_width) continue;

                // Scatter contribution to each output channel in the group
                for (int oc_rel = 0; oc_rel < out_channels_per_group; ++oc_rel) {
                    int oc = g * out_channels_per_group + oc_rel;

                    // Compute the index for the weight tensor for input channel ic
                    // Weight shape: [in_channels, out_channels/groups, kernel_h, kernel_w]
                    int weight_idx = ic * (out_channels_per_group * kernel_h * kernel_w) 
                                   + oc_rel * (kernel_h * kernel_w)
                                   + kh * kernel_w + kw;

                    scalar_t weight_val = weight[weight_idx];
                    scalar_t contrib = in_val * weight_val;

                    // Compute the global index for output [b, oc, oh, ow]
                    int output_idx = b * (out_channels * out_height * out_width)
                                   + oc * (out_height * out_width)
                                   + oh * out_width + ow;
                    
                    // Atomic add to safely accumulate contributions
                    atomicAdd(&output[output_idx], contrib);
                }
            }
        }
    }
}

// Forward function: prepares output (initialized with bias if provided) and launches the scatter kernel.

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

    // Weight shape assumed: [in_channels, out_channels/groups, kernel_h, kernel_w]
    const int out_channels = weight.size(1) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    // Compute output dimensions matching standard transposed convolution formula
    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    // Initialize output tensor. Preinitialize with bias if provided.
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    if (bias.has_value() && bias->defined()) {
        // Bias is 1D with out_channels elements; add to each output channel
        output = output + bias.value().view({1, -1, 1, 1});
    }

    // Launch the scatter kernel
    int total_threads = batch_size * in_channels * in_height * in_width;
    const int threads = 256;
    const int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_scatter_atomic_cuda", ([&] {
        conv_transpose2d_scatter_atomic_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
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
    m.def("forward", &forward, "Scatter Transposed 2D Convolution with Atomic Additions (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}
