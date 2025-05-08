#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel_unrolled(
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
    const int total_elements = batch_size * out_channels * out_height * out_width;
    
    #pragma unroll 2
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_elements; 
         idx += blockDim.x * gridDim.x) {
        
        // Unravel index
        const int ow = idx % out_width;
        const int oh = (idx / out_width) % out_height;
        const int oc = (idx / (out_width * out_height)) % out_channels;
        const int b = idx / (out_width * out_height * out_channels);

        const int out_channels_per_group = out_channels / groups;
        const int g = oc / out_channels_per_group;
        const int oc_group = oc % out_channels_per_group;
        const int in_channels_per_group = in_channels / groups;
        const int ic_start = g * in_channels_per_group;

        scalar_t val = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);

        // Manual partial unroll of kernel height loop for common kernel sizes
        #pragma unroll 4
        for (int kh = 0; kh < kernel_h; ++kh) {
            const int h_in_temp = oh - kh * dilation + padding;
            if (h_in_temp % stride != 0) continue;
            
            const int h_in = h_in_temp / stride;
            if (h_in < 0 || h_in >= in_height) continue;

            // Manual partial unroll of kernel width loop
            #pragma unroll 4
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int w_in_temp = ow - kw * dilation + padding;
                if (w_in_temp % stride != 0) continue;
                
                const int w_in = w_in_temp / stride;
                if (w_in < 0 || w_in >= in_width) continue;

                // Unroll the channel loop for better instruction-level parallelism
                #pragma unroll 4
                for (int ic = 0; ic < in_channels_per_group; ++ic) {
                    const int input_idx = b * (in_channels * in_height * in_width) +
                                        (ic_start + ic) * (in_height * in_width) +
                                        h_in * in_width + w_in;

                    const int weight_idx = (ic_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                                         oc_group * (kernel_h * kernel_w) +
                                         kh * kernel_w + kw;

                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }
        
        output[idx] = val;
    }
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
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = weight.size(1) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->numel() == out_channels, "Bias must have out_channels elements");
        TORCH_CHECK(bias->device().is_cuda(), "Bias must be a CUDA tensor");
    }

    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    const int total_elements = output.numel();
    constexpr int THREADS_PER_BLOCK = 256;
    const int num_blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda_unrolled", ([&] {
        conv_transpose2d_kernel_unrolled<scalar_t><<<num_blocks, THREADS_PER_BLOCK>>>(
            input.data_ptr<scalar_t>(),
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
    m.def("forward", &forward, "Unrolled Transposed 2D Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}