#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel focused on workload balancing to maximize GPU utilization

template <typename scalar_t>
__global__ void conv_transpose2d_kernel_balanced(
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
    // Calculate workload per thread
    int total_elements = batch_size * out_channels * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;

    // Loop over elements each thread will cover
    for (; idx < total_elements; idx += gridStride) {
        int n = idx;
        const int ow = n % out_width;
        n /= out_width;
        const int oh = n % out_height;
        n /= out_height;
        const int oc = n % out_channels;
        n /= out_channels;
        const int b = n;

        const int out_channels_per_group = out_channels / groups;
        const int g = oc / out_channels_per_group;
        const int oc_group = oc % out_channels_per_group;
        const int in_channels_per_group = in_channels / groups;
        const int ic_start = g * in_channels_per_group;

        scalar_t val = 0;
        if (bias != nullptr) {
            val = bias[oc];
        }

        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_in_idx = oh - kh * dilation + padding;
            if (h_in_idx % stride != 0) continue;
            int h_in = h_in_idx / stride;
            if (h_in < 0 || h_in >= in_height) continue;

            for (int kw = 0; kw < kernel_w; ++kw) {
                int w_in_idx = ow - kw * dilation + padding;
                if (w_in_idx % stride != 0) continue;
                int w_in = w_in_idx / stride;
                if (w_in < 0 || w_in >= in_width) continue;

                for (int ic = 0; ic < in_channels_per_group; ++ic) {
                    int input_idx = b * in_channels * in_height * in_width
                                   + (ic_start + ic) * in_height * in_width
                                   + h_in * in_width
                                   + w_in;

                    int weight_idx = (ic_start + ic) * out_channels_per_group * kernel_h * kernel_w
                                    + oc_group * kernel_h * kernel_w
                                    + kh * kernel_w
                                    + kw;

                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }

        output[idx] = val;
    }
}

// Launch kernel with appropriate blocks and threads

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

    const int out_channels = weight.size(1) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->numel() == out_channels, "Bias must have out_channels elements");
        TORCH_CHECK(bias->device().is_cuda(), "Bias must be a CUDA tensor");
    }

    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    const int total_elements = output.numel();

    constexpr int THREADS = 256;
    const int BLOCKS = min(65535, (total_elements + THREADS - 1) / THREADS);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda_balanced", ([&] {
        conv_transpose2d_kernel_balanced<scalar_t><<<BLOCKS, THREADS>>>(
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

// Pybind interface

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced Transposed 2D Convolution Kernel (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}