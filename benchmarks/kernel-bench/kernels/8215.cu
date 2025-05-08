#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
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
    extern __shared__ char shared_memory[];
    scalar_t* shared_bias = (scalar_t*)shared_memory;
    
    // Load bias into shared memory
    if (bias != nullptr && threadIdx.x < out_channels) {
        shared_bias[threadIdx.x] = bias[threadIdx.x];
    }
    __syncthreads();

    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Unravel index with improved arithmetic
    const int ow = idx % out_width;
    const int oh = (idx / out_width) % out_height;
    const int oc = (idx / (out_width * out_height)) % out_channels;
    const int b = idx / (out_width * out_height * out_channels);

    if (b >= batch_size) return;

    const int out_channels_per_group = out_channels / groups;
    const int g = oc / out_channels_per_group;
    const int oc_group = oc % out_channels_per_group;
    const int in_channels_per_group = in_channels / groups;
    const int ic_start = g * in_channels_per_group;

    scalar_t val = (bias != nullptr) ? shared_bias[oc] : static_cast<scalar_t>(0);

    // Pre-compute constant offsets
    const int input_batch_offset = b * in_channels * in_height * in_width;
    const int weight_group_offset = ic_start * (out_channels_per_group * kernel_h * kernel_w);
    const int oc_offset = oc_group * kernel_h * kernel_w;

    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        const int h_in = (oh - kh*dilation + padding) / stride;
        if ((oh - kh*dilation + padding) % stride != 0 || h_in < 0 || h_in >= in_height) continue;

        #pragma unroll
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int w_in = (ow - kw*dilation + padding) / stride;
            if ((ow - kw*dilation + padding) % stride != 0 || w_in < 0 || w_in >= in_width) continue;

            const int kernel_offset = kh * kernel_w + kw;

            for (int ic = 0; ic < in_channels_per_group; ++ic) {
                const int input_offset = input_batch_offset + 
                                        (ic_start + ic) * in_height * in_width + 
                                        h_in * in_width + 
                                        w_in;
                const int weight_offset = weight_group_offset + 
                                         ic * (out_channels_per_group * kernel_h * kernel_w) + 
                                         oc_offset + 
                                         kernel_offset;

                val += input[input_offset] * weight[weight_offset];
            }
        }
    }

    output[idx] = val;
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

    const int threads = 256;
    const int blocks = (output.numel() + threads - 1) / threads;
    

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
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
    m.def("forward", &forward, "Transposed 2D convolution (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}