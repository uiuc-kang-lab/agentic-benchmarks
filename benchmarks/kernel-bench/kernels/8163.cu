#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_gather_unrolled_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
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
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * out_height * out_width;
    if (idx >= total_elements) return;

    // Unravel output index
    const int ow = idx % out_width;
    const int oh = (idx / out_width) % out_height;
    const int oc = (idx / (out_width * out_height)) % out_channels;
    const int b = idx / (out_width * out_height * out_channels);

    // Precompute group-related indices
    const int out_channels_per_group = out_channels / groups;
    const int g = oc / out_channels_per_group;
    const int oc_group = oc % out_channels_per_group;
    const int in_channels_per_group = in_channels / groups;
    const int ic_start = g * in_channels_per_group;

    // Initialize output with bias
    scalar_t value = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);

    // Precompute constant offsets
    const int input_batch_stride = in_channels * in_height * in_width;
    const int input_channel_stride = in_height * in_width;
    const int weight_channel_stride = out_channels_per_group * kernel_h * kernel_w;
    const int weight_output_stride = kernel_h * kernel_w;

    // Main computation loops with unrolling
    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        const int h_offset = oh + padding - kh * dilation;
        if (h_offset < 0 || h_offset % stride != 0) continue;
        const int h_in = h_offset / stride;
        if (h_in >= in_height) continue;

        #pragma unroll
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int w_offset = ow + padding - kw * dilation;
            if (w_offset < 0 || w_offset % stride != 0) continue;
            const int w_in = w_offset / stride;
            if (w_in >= in_width) continue;

            // Process input channels in chunks of 4 when possible
            const int ic_aligned = (in_channels_per_group / 4) * 4;
            int ic = 0;

            // Manual unroll for chunks of 4 channels
            for (; ic < ic_aligned; ic += 4) {
                const int base_ic = ic_start + ic;
                
                // Compute input indices
                const int input_idx_base = b * input_batch_stride + h_in * in_width + w_in;
                const scalar_t in_val0 = input[input_idx_base + (base_ic) * input_channel_stride];
                const scalar_t in_val1 = input[input_idx_base + (base_ic + 1) * input_channel_stride];
                const scalar_t in_val2 = input[input_idx_base + (base_ic + 2) * input_channel_stride];
                const scalar_t in_val3 = input[input_idx_base + (base_ic + 3) * input_channel_stride];

                // Compute weight indices
                const int weight_idx_base = kh * kernel_w + kw;
                const scalar_t w_val0 = weight[base_ic * weight_channel_stride + oc_group * weight_output_stride + weight_idx_base];
                const scalar_t w_val1 = weight[(base_ic + 1) * weight_channel_stride + oc_group * weight_output_stride + weight_idx_base];
                const scalar_t w_val2 = weight[(base_ic + 2) * weight_channel_stride + oc_group * weight_output_stride + weight_idx_base];
                const scalar_t w_val3 = weight[(base_ic + 3) * weight_channel_stride + oc_group * weight_output_stride + weight_idx_base];

                // Accumulate products
                value += in_val0 * w_val0 + in_val1 * w_val1 + in_val2 * w_val2 + in_val3 * w_val3;
            }

            // Handle remaining channels
            for (; ic < in_channels_per_group; ++ic) {
                const int input_idx = b * input_batch_stride + (ic_start + ic) * input_channel_stride + 
                                    h_in * in_width + w_in;
                const int weight_idx = (ic_start + ic) * weight_channel_stride + 
                                     oc_group * weight_output_stride + 
                                     kh * kernel_w + kw;
                value += input[input_idx] * weight[weight_idx];
            }
        }
    }

    output[idx] = value;
}

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
    const int out_channels = weight.size(1) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    const int total_elements = output.numel();
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_gather_unrolled_cuda", ([&] {
        conv_transpose2d_gather_unrolled_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
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
    m.def("forward", &forward, "Unrolled Gather-based Transposed 2D Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}