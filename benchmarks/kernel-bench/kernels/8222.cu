#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define inline functions for clamping
__device__ inline int int_min(int a, int b) {
    return a < b ? a : b;
}

__device__ inline int int_max(int a, int b) {
    return a > b ? a : b;
}

// This device function computes the convolution value in a branchless manner.
// Instead of using conditional branches (if/continue), we compute a validity mask
// for the h and w indices and clamp the indices so that memory accesses are always safe.
// The invalid contributions are multiplied by 0, ensuring uniform control flow even when
// some kernel offsets fall outside the input region.

template <typename scalar_t>
__device__ scalar_t compute_conv_value(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const int b,
    const int oc,
    const int oh,
    const int ow,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int out_channels_per_group
) {
    const int in_channels_per_group = in_channels / groups;
    const int g = oc / out_channels_per_group;
    const int oc_group = oc % out_channels_per_group;
    const int ic_start = g * in_channels_per_group;
    
    scalar_t value = 0;
    
    // Loop over kernel height and width without divergent branches
    for (int kh = 0; kh < kernel_h; ++kh) {
        // Compute h offset and corresponding input index
        int h_offset = oh + padding - kh * dilation;
        int h_in = h_offset / stride;
        // Branchless validity: valid if remainder is 0 and h_in is in bounds
        int valid_h = ((h_offset - h_in * stride) == 0) * (h_in >= 0) * (h_in < in_height);
        // Clamp h_in safely so that the memory access is in-bound
        int h_safe = int_max(0, int_min(h_in, in_height - 1));
        
        for (int kw = 0; kw < kernel_w; ++kw) {
            int w_offset = ow + padding - kw * dilation;
            int w_in = w_offset / stride;
            int valid_w = ((w_offset - w_in * stride) == 0) * (w_in >= 0) * (w_in < in_width);
            int valid = valid_h * valid_w;  // 1 if both valid, else 0
            int w_safe = int_max(0, int_min(w_in, in_width - 1));
            
            // Accumulate contributions over the input channels for this kernel offset
            for (int ic = 0; ic < in_channels_per_group; ++ic) {
                int input_index = b * in_channels * in_height * in_width +
                                  (ic_start + ic) * in_height * in_width +
                                  h_safe * in_width + w_safe;
                int weight_index = (ic_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                                   oc_group * (kernel_h * kernel_w) +
                                   kh * kernel_w + kw;
                scalar_t x_val = input[input_index];
                scalar_t w_val = weight[weight_index];
                // Multiply by the validity flag to nullify contributions from invalid indices
                value += valid * x_val * w_val;
            }
        }
    }
    return value;
}

// The main kernel with uniform control flow. Each thread computes one output element. Use shared memory to optimize data access.
// Warps follow a uniform loop structure without divergent branches in the inner loops.

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
    const int groups,
    const int dilation,
    const int out_height,
    const int out_width
) {
    int total_elements = batch_size * out_channels * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Unravel the output index
    int ow = idx % out_width;
    int temp = idx / out_width;
    int oh = temp % out_height;
    temp = temp / out_height;
    int oc = temp % out_channels;
    int b = temp / out_channels;
    
    const int out_channels_per_group = out_channels / groups;
    scalar_t conv_val = compute_conv_value<scalar_t>(
                            input, weight, b, oc, oh, ow,
                            in_channels, in_height, in_width,
                            kernel_h, kernel_w, stride, padding,
                            dilation, groups, out_channels_per_group);
    
    // Add bias if provided
    if(bias != nullptr) {
        conv_val += bias[oc];
    }

    output[idx] = conv_val;
}

// The forward function that launches the kernel

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
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int out_channels = weight.size(1) * groups;  // Weight shape: [in_channels, out_channels/groups, kH, kW]
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    if(bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->numel() == out_channels, "Bias must have out_channels elements");
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
    }

    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());
    int total = output.numel();
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            (bias.has_value() && bias->defined()) ? bias->data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, in_height, in_width,
            out_channels,
            kernel_h, kernel_w,
            stride, padding, groups, dilation,
            out_height, out_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 2D convolution with minimized warp divergence (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}
