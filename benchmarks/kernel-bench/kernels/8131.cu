#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


// Kernel that uses shared memory to tile the weight matrix per group
// Grid is launched with 2D configuration: gridDim.y = groups; gridDim.x covers outputs for each group

template <typename scalar_t>
__global__ void conv_transpose2d_shared_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,       // equals groups * out_channels_per_group
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
    // Determine group-specific parameters
    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;

    // gridDim.y is set to 'groups'; each block in a y-slice processes outputs for that group
    const int g = blockIdx.y;  // current group index

    // Total number of output elements for this group
    const int group_output_elements = batch_size * out_channels_per_group * out_height * out_width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= group_output_elements) return;

    // Unravel idx into output coordinates within the group
    int temp = idx;
    const int ow = temp % out_width;
    temp /= out_width;
    const int oh = temp % out_height;
    temp /= out_height;
    const int oc_group = temp % out_channels_per_group;
    temp /= out_channels_per_group;
    const int b = temp;  // batch index

    // Global output channel index
    const int oc = g * out_channels_per_group + oc_group;
    const int global_out_idx = b * (out_channels * out_height * out_width) + 
                               oc * (out_height * out_width) + 
                               oh * out_width + 
                               ow;

    // Allocate shared memory for the weight tile of this group.
    // Weight shape global: [in_channels, out_channels_per_group, kernel_h, kernel_w]
    // For group g, we need the sub-tensor: [in_channels_per_group, out_channels_per_group, kernel_h, kernel_w]
    extern __shared__ char smem[];  
    scalar_t* s_weight = reinterpret_cast<scalar_t*>(smem);

    const int tile_size = in_channels_per_group * out_channels_per_group * kernel_h * kernel_w;
    // Base index in global weight for group g
    const int base_weight_idx = g * in_channels_per_group * (out_channels_per_group * kernel_h * kernel_w);

    // Each thread loads a portion of the weight tile into shared memory
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        s_weight[i] = weight[base_weight_idx + i];
    }
    // Synchronize to ensure the weight tile is fully loaded
    __syncthreads();

    // Initialize accumulator with bias if provided
    scalar_t val = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);

    // Loop over kernel height and width
    for (int kh = 0; kh < kernel_h; ++kh) {
        int h_offset = oh - kh * dilation + padding;
        if (h_offset % stride != 0) continue;
        int h_in = h_offset / stride;
        if (h_in < 0 || h_in >= in_height) continue;

        for (int kw = 0; kw < kernel_w; ++kw) {
            int w_offset = ow - kw * dilation + padding;
            if (w_offset % stride != 0) continue;
            int w_in = w_offset / stride;
            if (w_in < 0 || w_in >= in_width) continue;

            // Loop over the input channels within the current group
            for (int ic = 0; ic < in_channels_per_group; ++ic) {
                // Global channel index: g * in_channels_per_group + ic
                int input_idx = b * (in_channels * in_height * in_width) +
                                (g * in_channels_per_group + ic) * (in_height * in_width) +
                                h_in * in_width + w_in;
                scalar_t x_val = input[input_idx];

                // Access preloaded weight from shared memory
                int weight_idx = ic * (out_channels_per_group * kernel_h * kernel_w) +
                                 oc_group * (kernel_h * kernel_w) +
                                 kh * kernel_w + kw;
                scalar_t w_val = s_weight[weight_idx];
                val += x_val * w_val;
            }
        }
    }

    output[global_out_idx] = val;
}


// Forward function using the optimized kernel

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

    // Weight shape: [in_channels, out_channels_per_group, kernel_h, kernel_w]
    const int out_channels_per_group = weight.size(1);
    const int out_channels = out_channels_per_group * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    // Validate bias if provided
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->numel() == out_channels, "Bias must have out_channels elements");
        TORCH_CHECK(bias->device().is_cuda(), "Bias must be a CUDA tensor");
    }

    // Calculate output dimensions
    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    // For the optimized kernel, we launch a 2D grid:
    // gridDim.y corresponds to groups
    // For each group, total outputs: batch_size * out_channels_per_group * out_height * out_width
    const int group_output_elements = batch_size * out_channels_per_group * out_height * out_width;

    const int threads = 256;
    const int blocks_x = (group_output_elements + threads - 1) / threads;
    dim3 blocks(blocks_x, groups);

    // Shared memory size for loading weight tile per group
    const int in_channels_per_group = in_channels / groups;
    size_t smem_size = sizeof(float) * in_channels_per_group * out_channels_per_group * kernel_h * kernel_w;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_shared_cuda", ([&] {
        conv_transpose2d_shared_kernel<scalar_t><<<blocks, threads, smem_size>>>(
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
    m.def("forward", &forward, "Transposed 2D convolution with shared memory (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}
