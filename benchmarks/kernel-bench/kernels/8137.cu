#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel_2d(
    const scalar_t* input,
    const scalar_t* weight,
    const scalar_t* bias,
    scalar_t* output,
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
    // 2D block coordinates
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Calculate output position
    const int oh = by * blockDim.y + ty;
    const int ow = bx * blockDim.x + tx;
    
    // Early exit if outside output dimensions
    if (oh >= out_height || ow >= out_width) return;

    // Process multiple channels/batches per thread block in z-dimension
    const int items_per_z = (batch_size * out_channels + gridDim.z - 1) / gridDim.z;
    const int z_start = bz * items_per_z;
    const int z_end = min(z_start + items_per_z, batch_size * out_channels);

    for (int idx = z_start; idx < z_end; idx++) {
        const int b = idx / out_channels;
        const int oc = idx % out_channels;

        if (b >= batch_size) continue;

        const int out_channels_per_group = out_channels / groups;
        const int g = oc / out_channels_per_group;
        const int oc_group = oc % out_channels_per_group;
        const int in_channels_per_group = in_channels / groups;
        const int ic_start = g * in_channels_per_group;

        scalar_t val = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);

        #pragma unroll 4
        for (int kh = 0; kh < kernel_h; ++kh) {
            const int h_in = (oh - kh * dilation + padding) / stride;
            const bool valid_h = (h_in >= 0 && h_in < in_height && 
                                (oh - kh * dilation + padding) % stride == 0);

            if (!valid_h) continue;

            #pragma unroll 4
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int w_in = (ow - kw * dilation + padding) / stride;
                const bool valid_w = (w_in >= 0 && w_in < in_width && 
                                    (ow - kw * dilation + padding) % stride == 0);

                if (!valid_w) continue;

                for (int ic = 0; ic < in_channels_per_group; ++ic) {
                    const scalar_t x_val = input[((b * in_channels + (ic_start + ic)) * in_height + h_in) * in_width + w_in];
                    const scalar_t w_val = weight[((ic_start + ic) * out_channels_per_group + oc_group) * kernel_h * kernel_w + 
                                                 kh * kernel_w + kw];
                    val += x_val * w_val;
                }
            }
        }

        const int out_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
        output[out_idx] = val;
    }
}

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

    // 2D block configuration
    dim3 threads(16, 16);
    dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        min(32, (batch_size * out_channels + 31) / 32) // Z-dimension for batch/channel processing
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda_2d", ([&] {
        conv_transpose2d_kernel_2d<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &forward, "Transposed 2D convolution with 2D block mapping (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}