#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile sizes for better cache utilization
#define TILE_SIZE 16
#define BLOCK_SIZE 256

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
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
    // Calculate tile indices
    const int tile_row = blockIdx.y * TILE_SIZE;
    const int tile_col = blockIdx.x * TILE_SIZE;
    const int batch_idx = blockIdx.z / out_channels;
    const int out_channel = blockIdx.z % out_channels;

    // Thread indices within the block
    const int thread_row = threadIdx.x / TILE_SIZE;
    const int thread_col = threadIdx.x % TILE_SIZE;

    // Calculate actual output position
    const int oh = tile_row + thread_row;
    const int ow = tile_col + thread_col;

    if (batch_idx >= batch_size || oh >= out_height || ow >= out_width) return;

    // Group and channel computations
    const int out_channels_per_group = out_channels / groups;
    const int g = out_channel / out_channels_per_group;
    const int oc_group = out_channel % out_channels_per_group;
    const int in_channels_per_group = in_channels / groups;
    const int ic_start = g * in_channels_per_group;

    // Pre-compute strides
    const int input_b_stride = in_channels * in_height * in_width;
    const int input_c_stride = in_height * in_width;
    const int weight_channel_stride = kernel_h * kernel_w;

    // Initialize accumulator
    scalar_t val = (bias != nullptr) ? bias[out_channel] : static_cast<scalar_t>(0);

    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        const int h_in_base = oh - kh * dilation + padding;
        const int h_in = h_in_base / stride;
        
        if (h_in_base % stride == 0 && h_in >= 0 && h_in < in_height) {
            #pragma unroll
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int w_in_base = ow - kw * dilation + padding;
                const int w_in = w_in_base / stride;
                
                if (w_in_base % stride == 0 && w_in >= 0 && w_in < in_width) {
                    #pragma unroll 4
                    for (int ic = 0; ic < in_channels_per_group; ++ic) {
                        const int input_idx = batch_idx * input_b_stride + 
                                            (ic_start + ic) * input_c_stride + 
                                            h_in * in_width + w_in;
                        
                        const int weight_idx = (ic_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                                             oc_group * weight_channel_stride +
                                             kh * kernel_w + kw;
                        
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Write output
    if (oh < out_height && ow < out_width) {
        const int out_idx = batch_idx * out_channels * out_height * out_width +
                           out_channel * out_height * out_width +
                           oh * out_width + ow;
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

    // Calculate grid dimensions for tiled approach
    dim3 grid(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );
    dim3 block(BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<grid, block>>>(
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
    m.def("forward", &forward, "Transposed 2D convolution with tiled distribution (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}