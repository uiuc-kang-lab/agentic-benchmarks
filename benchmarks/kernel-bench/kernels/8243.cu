#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This function calculates the 4D output indices for a flattened index
template <typename scalar_t>
__device__ void calculate_indices(
    int idx,
    const int out_width,
    const int out_height,
    const int out_channels,
    int &b,
    int &oc,
    int &oh,
    int &ow
) {
    int n = idx;
    ow = n % out_width;
    n /= out_width;
    oh = n % out_height;
    n /= out_height;
    oc = n % out_channels;
    n /= out_channels;
    b = n;
}

// Branchless computation of valid mask: returns 1 if condition true, 0 otherwise
// Avoiding divergent branches by using arithmetic (the ternary operator
// is typically compiled to a predicated instruction on NVIDIA GPUs)

__device__ __forceinline__ int valid_mask(int condition) {
    // condition is either true (nonzero) or false (0).
    // We use !! to normalize to 1 or 0.
    return !!condition;
}

// Compute the convolution contribution for one output element without divergent branches
// All conditional checks in the inner loops are computed as arithmetic masks
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
    // Determine group and channel offsets
    const int g = oc / out_channels_per_group;
    const int oc_group = oc % out_channels_per_group; 
    const int in_channels_per_group = in_channels / groups;
    const int ic_start = g * in_channels_per_group;

    scalar_t sum = 0;

    // Loop over kernel height and width
    for (int kh = 0; kh < kernel_h; ++kh) {
        // Compute h offset and index with branchless validity check
        int h_offset = oh - kh * dilation + padding;
        int h_in = h_offset / stride;  // integer division
        int valid_h = valid_mask((h_offset % stride) == 0) * valid_mask((h_in >= 0) && (h_in < in_height));
        
        for (int kw = 0; kw < kernel_w; ++kw) {
            int w_offset = ow - kw * dilation + padding;
            int w_in = w_offset / stride;
            int valid_w = valid_mask((w_offset % stride) == 0) * valid_mask((w_in >= 0) && (w_in < in_width));
            int valid = valid_h * valid_w;
            
            // Iterate over input channels within the group
            for (int ic = 0; ic < in_channels_per_group; ++ic) {
                int input_idx = b * in_channels * in_height * in_width +
                                (ic_start + ic) * in_height * in_width +
                                h_in * in_width + w_in;
                int weight_idx = (ic_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                                 oc_group * (kernel_h * kernel_w) +
                                 kh * kernel_w + kw;
                // Multiply by the valid mask so that invalid contributions add 0
                scalar_t x_val = input[input_idx];
                scalar_t w_val = weight[weight_idx];
                sum += x_val * w_val * valid;
            }
        }
    }
    return sum;
}

// Main CUDA kernel for transposed 2D convolution with branchless inner loops
// to minimize warp divergence
template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
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
    const int groups,
    const int dilation,
    const int out_height,
    const int out_width
) {
    const int total_elements = batch_size * out_channels * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int b, oc, oh, ow;
    calculate_indices<scalar_t>(idx, out_width, out_height, out_channels, b, oc, oh, ow);

    const int out_channels_per_group = out_channels / groups;
    scalar_t val = compute_conv_value<scalar_t>(
        input, weight, b, oc, oh, ow,
        in_channels, in_height, in_width,
        kernel_h, kernel_w, stride, padding,
        dilation, groups, out_channels_per_group
    );

    // Add bias; since the presence of bias is uniform across the kernel launch,
    // the same control flow applies for all threads.
    val += (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);

    output[idx] = val;
}

// The forward function to launch the CUDA kernel

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
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    // out_channels = weight.size(1) * groups; weight shape: [in_channels, out_channels/groups, kH, kW]
    const int out_channels = weight.size(1) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->numel() == out_channels, "Bias must have out_channels elements");
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
    }

    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    const int total_elements = output.numel();
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
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
            groups,
            dilation,
            out_height,
            out_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 2D convolution with branchless inner loops (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}
