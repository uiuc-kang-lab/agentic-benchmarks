#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel_strided(
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
    // Calculate stride parameters for efficient memory access
    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;
    const int elements_per_thread = 4; // Process multiple elements per thread
    const int total_elements = batch_size * out_channels * out_height * out_width;
    
    // Grid-stride loop with multiple elements per iteration
    for (int base_idx = blockIdx.x * blockDim.x + threadIdx.x; 
         base_idx < total_elements; 
         base_idx += blockDim.x * gridDim.x * elements_per_thread) {
        
        #pragma unroll
        for (int offset = 0; offset < elements_per_thread; offset++) {
            const int idx = base_idx + offset * blockDim.x * gridDim.x;
            if (idx >= total_elements) continue;

            // Compute 4D indices
            const int ow = idx % out_width;
            const int oh = (idx / out_width) % out_height;
            const int oc = (idx / (out_width * out_height)) % out_channels;
            const int b = idx / (out_width * out_height * out_channels);

            // Group-related calculations
            const int g = oc / out_channels_per_group;
            const int oc_group = oc % out_channels_per_group;
            const int ic_start = g * in_channels_per_group;

            // Initialize output value
            scalar_t sum = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);

            // Compute valid ranges for kernel window
            const int kh_start = (oh + padding) / stride;
            const int kw_start = (ow + padding) / stride;
            const int kh_end = min(kernel_h, (oh + padding + stride - 1) / stride);
            const int kw_end = min(kernel_w, (ow + padding + stride - 1) / stride);

            // Process kernel window with optimized bounds
            #pragma unroll 4
            for (int kh = kh_start; kh < kh_end; kh++) {
                const int h_in = (oh - kh * dilation + padding) / stride;
                if (h_in < 0 || h_in >= in_height) continue;

                #pragma unroll 4
                for (int kw = kw_start; kw < kw_end; kw++) {
                    const int w_in = (ow - kw * dilation + padding) / stride;
                    if (w_in < 0 || w_in >= in_width) continue;

                    // Process input channels with vectorized loads where possible
                    for (int ic = 0; ic < in_channels_per_group; ic++) {
                        const int input_idx = ((b * in_channels + (ic_start + ic)) * in_height + h_in) * in_width + w_in;
                        const int weight_idx = ((ic_start + ic) * out_channels_per_group + oc_group) * (kernel_h * kernel_w) 
                                             + kh * kernel_w + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
            
            output[idx] = sum;
        }
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
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / (threads_per_block * 4); // Adjust for elements_per_thread

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda_strided", ([&] {
        conv_transpose2d_kernel_strided<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &forward, "Strided Transposed 2D Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}