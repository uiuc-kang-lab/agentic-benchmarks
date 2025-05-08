#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// New kernel using grid-stride loop for improved load balancing across threads

template <typename scalar_t>
__global__ void conv_transpose2d_kernel_gridstride(
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
    
    // Grid-stride loop to distribute workload evenly
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
        int n = idx;
        const int ow = n % out_width;
        n /= out_width;
        const int oh = n % out_height;
        n /= out_height;
        const int oc = n % out_channels;
        n /= out_channels;
        const int b = n;
        
        // Determine group and channel offset
        const int out_channels_per_group = out_channels / groups;
        const int g = oc / out_channels_per_group;
        const int oc_group = oc % out_channels_per_group;
        const int in_channels_per_group = in_channels / groups;
        const int ic_start = g * in_channels_per_group;
        
        // Initialize accumulation with bias if provided
        scalar_t val = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);
        
        // Loop over kernel dimensions
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_offset = oh - kh * dilation + padding;
                if (h_offset < 0 || h_offset % stride != 0) continue;
                int h_in = h_offset / stride;
                if (h_in < 0 || h_in >= in_height) continue;
                
                int w_offset = ow - kw * dilation + padding;
                if (w_offset < 0 || w_offset % stride != 0) continue;
                int w_in = w_offset / stride;
                if (w_in < 0 || w_in >= in_width) continue;
                
                // Accumulate over the corresponding input channels
                for (int ic = 0; ic < in_channels_per_group; ++ic) {
                    const int input_idx = b * in_channels * in_height * in_width
                                            + (ic_start + ic) * in_height * in_width
                                            + h_in * in_width
                                            + w_in;
                    const scalar_t x_val = input[input_idx];
                    
                    const int weight_idx = (ic_start + ic) * (out_channels_per_group * kernel_h * kernel_w)
                                             + oc_group * (kernel_h * kernel_w)
                                             + kh * kernel_w + kw;
                    const scalar_t w_val = weight[weight_idx];
                    
                    val += x_val * w_val;
                }
            }
        }
        output[idx] = val;
    }
}

// Forward function wrapping the CUDA kernel launch

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

    // Compute the number of output channels based on groups
    const int out_channels = weight.size(1) * groups;  // weight shape: [in_channels, out_channels/groups, kH, kW]
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->numel() == out_channels, "Bias must have out_channels elements");
        TORCH_CHECK(bias->device().is_cuda(), "Bias must be a CUDA tensor");
    }

    // Calculate output spatial dimensions
    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    const int total_elements = output.numel();
    const int threads = 1024;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel_gridstride<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &forward, "Transposed 2D convolution with grid-stride loop (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}
