#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


// This kernel processes a tile of the output corresponding to one group and one batch.
// It preloads the weight tile for the group into shared memory using a single __syncthreads(),
// ensuring that shared memory is used only where needed and synchronizations are minimal.


template <typename scalar_t>
__global__ void conv_transpose2d_kernel_shared(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels, // = groups * out_channels_per_group
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
    // Determine batch and group from grid indices
    int b = blockIdx.z;        // batch index
    int g = blockIdx.y;        // group index

    // Calculate per-group channel counts
    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;

    // Load the weight tile for group g into shared memory
    // Weight shape is [in_channels, out_channels/group, kernel_h, kernel_w]
    // For group g, we load weights for channels from g*in_channels_per_group to (g+1)*in_channels_per_group - 1
    // Total elements to load:
    int weight_tile_size = in_channels_per_group * out_channels_per_group * kernel_h * kernel_w;

    extern __shared__ char smem[]; // raw shared memory
    scalar_t* shared_weight = reinterpret_cast<scalar_t*>(smem);

    // Use blockDim.x threads to cooperatively load the weight tile from global memory
    for (int i = threadIdx.x; i < weight_tile_size; i += blockDim.x) {
        // Global weight index: each group starts at offset = g * in_channels_per_group * (out_channels_per_group * kernel_h * kernel_w)
        int global_idx = g * in_channels_per_group * (out_channels_per_group * kernel_h * kernel_w) + i;
        shared_weight[i] = weight[global_idx];
    }
    __syncthreads(); // synchronize to ensure shared_weight is fully loaded

    // Now, each block processes a portion of the output for this (b, g) tile.
    // The output tile for this group has size: out_channels_per_group * out_height * out_width
    int out_tile_size = out_channels_per_group * out_height * out_width;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= out_tile_size) return;

    // Unravel tid to get local output channel, output height and width indices
    int oc_local = tid / (out_height * out_width);
    int residual = tid % (out_height * out_width);
    int oh = residual / out_width;
    int ow = residual % out_width;

    // Actual output channel index in full tensor
    int oc = g * out_channels_per_group + oc_local;
    
    // Initialize the output value with bias if provided
    scalar_t out_val = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);

    // Compute the convolution sum
    // Loop over the kernel spatial dimensions
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            // Compute corresponding input coordinate for this kernel position
            int h_tmp = oh - kh * dilation + padding;
            if (h_tmp < 0 || (h_tmp % stride) != 0) continue;
            int h_in = h_tmp / stride;
            if (h_in < 0 || h_in >= in_height) continue;

            int w_tmp = ow - kw * dilation + padding;
            if (w_tmp < 0 || (w_tmp % stride) != 0) continue;
            int w_in = w_tmp / stride;
            if (w_in < 0 || w_in >= in_width) continue;

            // Loop over the input channels in this group
            for (int ic = 0; ic < in_channels_per_group; ++ic) {
                // Compute index into input tensor: shape [batch, in_channels, in_height, in_width]
                int input_idx = b * (in_channels * in_height * in_width)
                              + (g * in_channels_per_group + ic) * (in_height * in_width)
                              + h_in * in_width + w_in;
                scalar_t in_val = input[input_idx];

                // Get the corresponding weight from shared memory
                // Weight tile layout: [in_channels_per_group, out_channels_per_group, kernel_h, kernel_w]
                // Index: ic * (out_channels_per_group * kernel_h * kernel_w) + oc_local * (kernel_h * kernel_w) + (kh * kernel_w + kw)
                int w_idx = ic * (out_channels_per_group * kernel_h * kernel_w)
                          + oc_local * (kernel_h * kernel_w)
                          + (kh * kernel_w + kw);
                scalar_t w_val = shared_weight[w_idx];

                out_val += in_val * w_val;
            }
        }
    }

    // Write the computed value to the output tensor: shape [batch, out_channels, out_height, out_width]
    int output_idx = b * (out_channels * out_height * out_width)
                   + oc * (out_height * out_width)
                   + oh * out_width + ow;
    output[output_idx] = out_val;
}


// Forward function that sets up kernel launch configuration

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

    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->numel() == out_channels, "Bias must have out_channels elements");
        TORCH_CHECK(bias->device().is_cuda(), "Bias must be a CUDA tensor");
    }

    // Calculate output dimensions
    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    // Configure grid dimensions for our 3D grid: x covers the output tile, y covers groups, z covers batch
    int in_channels_per_group = in_channels / groups;
    int out_tile_size = out_channels_per_group * out_height * out_width; // number of output elements per (batch, group)

    // Choose a block size (number of threads per block) for processing the output tile
    constexpr int BLOCK_SIZE = 256;
    int grid_x = (out_tile_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(grid_x, groups, batch_size);
    dim3 block(BLOCK_SIZE);

    // Calculate shared memory size needed for one weight tile for a group
    size_t shared_mem_size = in_channels_per_group * out_channels_per_group * kernel_h * kernel_w * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda_opt_shared", ([&] {
        conv_transpose2d_kernel_shared<scalar_t><<<grid, block, shared_mem_size>>>(
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
