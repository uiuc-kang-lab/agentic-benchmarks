#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined kernel with warp-level shfl reduction and memory coalescing optimization

template <typename scalar_t>
__global__ void transposed_conv3d_optimized_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,  // can be nullptr
    scalar_t* __restrict__ output,
    // Input dimensions
    int N, int in_channels, int in_depth, int in_height, int in_width,
    // Output dimensions
    int out_channels, int out_depth, int out_height, int out_width,
    // Kernel dimensions
    int kT, int kH, int kW,
    // Stride
    int stride_d, int stride_h, int stride_w,
    // Padding
    int pad_d, int pad_h, int pad_w,
    // Output padding
    int out_pad_d, int out_pad_h, int out_pad_w,
    // Groups
    int groups
) {
    const int warpSize = 32;
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warpSize;  // one warp per output element
    int lane = global_thread_id % warpSize;

    int total_output = N * out_channels * out_depth * out_height * out_width;
    if (warp_id >= total_output) return;

    // Decode warp_id to (n, c, d, h, w) for the output element
    int w_out = warp_id % out_width;
    int tmp = warp_id / out_width;
    int h_out = tmp % out_height;
    tmp /= out_height;
    int d_out = tmp % out_depth;
    tmp /= out_depth;
    int c_out = tmp % out_channels;
    int n = tmp / out_channels;

    // Determine group assignment and local channel index
    int group = c_out / (out_channels / groups);
    int out_c_local = c_out % (out_channels / groups);

    // Each group has a subset of input channels
    int in_channels_per_group = in_channels / groups;
    // Total iterations: over input channels for this group and kernel volume
    int total_iters = in_channels_per_group * (kT * kH * kW);

    scalar_t sum = 0;
    // Each lane processes a subset of the iterations
    for (int i = lane; i < total_iters; i += warpSize) {
        int ic = i / (kT * kH * kW);
        int rem = i % (kT * kH * kW);
        int kd = rem / (kH * kW);
        int rem2 = rem % (kH * kW);
        int kh = rem2 / kW;
        int kw = rem2 % kW;

        int input_channel = group * in_channels_per_group + ic;

        // Compute the corresponding input indices based on transposed convolution arithmetic
        int d_in_tmp = d_out + pad_d - kd;
        if (d_in_tmp % stride_d != 0) continue;
        int d_in = d_in_tmp / stride_d;
        if (d_in < 0 || d_in >= in_depth) continue;

        int h_in_tmp = h_out + pad_h - kh;
        if (h_in_tmp % stride_h != 0) continue;
        int h_in = h_in_tmp / stride_h;
        if (h_in < 0 || h_in >= in_height) continue;

        int w_in_tmp = w_out + pad_w - kw;
        if (w_in_tmp % stride_w != 0) continue;
        int w_in = w_in_tmp / stride_w;
        if (w_in < 0 || w_in >= in_width) continue;

        // Compute flat index in the input tensor
        int input_idx = (((n * in_channels + input_channel) * in_depth + d_in) * in_height + h_in) * in_width + w_in;

        // Compute weight index: weight shape is [in_channels, out_channels/groups, kT, kH, kW]
        int weight_idx = ((((input_channel) * (out_channels / groups) + out_c_local) * kT + kd) * kH + kh) * kW + kw;

        // Use __ldg for read-only cache optimization
        sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
    }

    // Use warp-level reduction with __shfl_down_sync to sum partial results
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        if (bias != nullptr) {
            sum += __ldg(&bias[c_out]);
        }
        output[warp_id] = sum;
    }
}

// Host function to launch the kernel

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Ensure inputs are contiguous
    input = input.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_tensor;
    if (bias.has_value()) {
        bias_tensor = bias.value().contiguous();
    }

    int N = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int kT = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    // Derive out_channels from weight and groups
    int out_channels = weight.size(1) * groups;
    int out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kT + output_padding[0];
    int out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    int out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];

    auto output = torch::zeros({N, out_channels, out_depth, out_height, out_width}, input.options());

    // Each warp computes one output element
    int total_output = N * out_channels * out_depth * out_height * out_width;
    const int warpSize = 32;
    int total_threads = total_output * warpSize;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_optimized_kernel", ([&] {
        transposed_conv3d_optimized_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias_tensor.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            N, in_channels, in_depth, in_height, in_width,
            out_channels, out_depth, out_height, out_width,
            kT, kH, kW,
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            output_padding[0], output_padding[1], output_padding[2],
            groups
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized ConvTranspose3d forward function",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}
