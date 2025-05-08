#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel optimizing with shared memory usage to reduce global memory access

template <typename scalar_t>
__global__ void conv_transpose2d_kernel_shared(
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
    extern __shared__ scalar_t sdata[];  // Shared memory for weights

    const int out_channels_per_group = out_channels / groups;

    const int weight_size = in_channels * out_channels / groups * kernel_h * kernel_w;
    scalar_t* shared_weights = sdata;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += gridStride) {
        int n = idx;
        const int ow = n % out_width;
        n /= out_width;
        const int oh = n % out_height;
        n /= out_height;
        const int oc = n % out_channels;
        n /= out_channels;
        const int b = n;

        const int g = oc / out_channels_per_group;
        const int oc_group = oc % out_channels_per_group;
        const int in_channels_per_group = in_channels / groups;
        const int ic_start = g * in_channels_per_group;

        scalar_t val = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);

        for (int ic = 0; ic < in_channels_per_group; ++ic) {
            const int ic_global = ic_start + ic;
            const int weight_idx = ic_global * out_channels_per_group * kernel_h * kernel_w + 
                                    oc_group * kernel_h * kernel_w;
            shared_weights[threadIdx.x] = weight[weight_idx];
            __syncthreads();

            for (int kh = 0; kh < kernel_h; ++kh) {
                int h_in = (oh - kh * dilation + padding) / stride;
                bool valid_h = h_in >= 0 && h_in < in_height && (oh - kh * dilation + padding) % stride == 0;
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int w_in = (ow - kw * dilation + padding) / stride;
                    bool valid_w = w_in >= 0 && w_in < in_width && (ow - kw * dilation + padding) % stride == 0;
                    if (valid_h && valid_w) {
                        const scalar_t x_val = input[b * in_channels * in_height * in_width
                                                  + ic_global * in_height * in_width
                                                  + h_in * in_width + w_in];

                        const scalar_t w_val = shared_weights[kh * kernel_w + kw];
                        val += x_val * w_val;
                    }
                }
            }
            __syncthreads();
        }

        output[idx] = val;
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

    const int total_elements = output.numel();

    constexpr int BLOCK_SIZE = 256;
    const int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Calculate shared memory size: allow enough space for sharing weight data
    const size_t shared_mem_size = weight_size * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda_shared", ([&] {
        conv_transpose2d_kernel_shared<scalar_t><<<blocks, BLOCK_SIZE, shared_mem_size>>>(
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
    m.def("forward", &forward, "Optimized Transposed 2D Convolution with Shared Memory (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}
