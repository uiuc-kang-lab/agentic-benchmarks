#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Scatter-based transposed convolution kernel using atomicAdd
// Each thread processes one input element and scatters its contributions
// to output using atomicAdd to avoid race conditions.

template <typename scalar_t>
__global__ void transposed_conv3d_scatter_atomic_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    // Input dimensions
    int N, int in_channels, int in_depth, int in_height, int in_width,
    // Output dimensions
    int out_channels, int out_depth, int out_height, int out_width,
    // Kernel dimensions
    int kT, int kH, int kW,
    // Strides
    int stride_d, int stride_h, int stride_w,
    // Padding
    int pad_d, int pad_h, int pad_w,
    // Groups
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_input = N * in_channels * in_depth * in_height * in_width;
    if (idx >= total_input) return;

    // Decode input index: [N, in_channels, in_depth, in_height, in_width]
    int w_in = idx % in_width;
    int tmp = idx / in_width;
    int h_in = tmp % in_height;
    tmp /= in_height;
    int d_in = tmp % in_depth;
    tmp /= in_depth;
    int c_in = tmp % in_channels;
    int n = tmp / in_channels;

    // Determine group and corresponding out_channels per group
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;  // weight shape: [in_channels, out_channels_per_group, kT, kH, kW]
    int group = c_in / in_channels_per_group;

    scalar_t input_val = input[idx];

    // Scatter contributions for each kernel element
    for (int kd = 0; kd < kT; kd++) {
        int out_d = d_in * stride_d - pad_d + kd;
        if (out_d < 0 || out_d >= out_depth) continue;
        for (int kh = 0; kh < kH; kh++) {
            int out_h = h_in * stride_h - pad_h + kh;
            if (out_h < 0 || out_h >= out_height) continue;
            for (int kw = 0; kw < kW; kw++) {
                int out_w = w_in * stride_w - pad_w + kw;
                if (out_w < 0 || out_w >= out_width) continue;
                // For each output channel in the current group
                for (int oc = 0; oc < out_channels_per_group; oc++) {
                    int out_c = group * out_channels_per_group + oc;
                    // Compute weight index: weight shape [in_channels, out_channels_per_group, kT, kH, kW]
                    int weight_idx = (((c_in * out_channels_per_group + oc) * kT + kd) * kH + kh) * kW + kw;
                    scalar_t w_val = weight[weight_idx];
                    scalar_t contribution = input_val * w_val;

                    // Compute output index: output shape [N, out_channels, out_depth, out_height, out_width]
                    int out_idx = (((n * out_channels + out_c) * out_depth + out_d) * out_height + out_h) * out_width + out_w;
                    atomicAdd(&output[out_idx], contribution);
                }
            }
        }
    }
}

// Kernel to add bias to the output tensor
// Each thread processes one output element and adds the corresponding bias

template <typename scalar_t>
__global__ void add_bias_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ bias,
    int total, int out_channels, int out_depth, int out_height, int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Compute channel index for the output element
    int spatial = out_depth * out_height * out_width;
    int c = (idx / spatial) % out_channels;
    output[idx] += bias[c];
}


// Host function for the scatter atomic optimized transposed convolution

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Ensure contiguous tensors
    input = input.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_tensor;
    if (bias.has_value()) {
        bias_tensor = bias.value().contiguous();
    }

    // Input dimensions: [N, in_channels, in_depth, in_height, in_width]
    int N = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    // Kernel dimensions: weight shape [in_channels, out_channels_per_group, kT, kH, kW]
    int kT = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    // Determine out_channels from weight and groups
    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;

    // Compute output dimensions using the transposed convolution formula:
    // out_dim = (in_dim - 1) * stride - 2 * padding + kernel_size + output_padding
    int out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kT + output_padding[0];
    int out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    int out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];

    // Allocate output tensor and initialize to zero
    auto output = torch::zeros({N, out_channels, out_depth, out_height, out_width}, input.options());

    // Launch scatter atomic kernel: map threads to input elements
    int total_input = N * in_channels * in_depth * in_height * in_width;
    int threads = 256;
    int blocks = (total_input + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_scatter_atomic_kernel", ([&] {
        transposed_conv3d_scatter_atomic_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, in_channels, in_depth, in_height, in_width,
            out_channels, out_depth, out_height, out_width,
            kT, kH, kW,
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            groups
        );
    }));

    // If bias is provided, add it in a separate kernel to minimize atomic overhead
    if (bias.has_value()) {
        int total_output = N * out_channels * out_depth * out_height * out_width;
        int threads_bias = 256;
        int blocks_bias = (total_output + threads_bias - 1) / threads_bias;
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "add_bias_kernel", ([&] {
            add_bias_kernel<scalar_t><<<blocks_bias, threads_bias>>>(
                output.data_ptr<scalar_t>(),
                bias_tensor.data_ptr<scalar_t>(),
                total_output, out_channels, out_depth, out_height, out_width
            );
        }));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Scatter-based Atomic Optimized ConvTranspose3d forward function",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}
