#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Inline device function to compute quotient and validity in a branchless manner.
// It computes the safe index (i.e. input coordinate) and returns 1 if valid, else 0.
__device__ inline int compute_index_and_valid(int offset, int stride, int size, int &safe_idx) {
    // Compute quotient and remainder
    int idx = offset / stride;
    int r = offset - idx * stride;  // equivalent to offset % stride
    // Determine validity: remainder must be 0 and index within bounds
    int valid = ((r == 0) && (idx >= 0) && (idx < size)) ? 1 : 0;
    // If not valid, use 0 as safe index to avoid out-of-bound access
    safe_idx = valid * idx;
    return valid;
}


// Branchless kernel for ConvTranspose3d that minimizes warp divergence
// by refactoring conditional logic into arithmetic masks.

template <typename scalar_t>
__global__ void transposed_conv3d_branchless_kernel(
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
    // Strides
    int stride_d, int stride_h, int stride_w,
    // Padding
    int pad_d, int pad_h, int pad_w,
    // Output padding (unused in computation below)
    int out_pad_d, int out_pad_h, int out_pad_w,
    // Groups
    int groups
) {
    int total = N * out_channels * out_depth * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decode flat index into (n, c, d, h, w)
    int w = idx % out_width;
    int tmp = idx / out_width;
    int h = tmp % out_height;
    tmp /= out_height;
    int d = tmp % out_depth;
    tmp /= out_depth;
    int c = tmp % out_channels;
    int n = tmp / out_channels;

    // Determine group information
    int out_channels_per_group = out_channels / groups;
    int group = c / out_channels_per_group;
    int out_c_local = c % out_channels_per_group;

    scalar_t sum = 0;
    int in_channels_per_group = in_channels / groups;

    // Loop over input channels in this group
    for (int ic = 0; ic < in_channels_per_group; ic++) {
        int input_channel = group * in_channels_per_group + ic;
        // Loop over kernel depth
        for (int kd = 0; kd < kT; kd++) {
            int d_offset = d + pad_d - kd;
            int d_in_safe;
            int valid_d = compute_index_and_valid(d_offset, stride_d, in_depth, d_in_safe);
            // Loop over kernel height
            for (int kh = 0; kh < kH; kh++) {
                int h_offset = h + pad_h - kh;
                int h_in_safe;
                int valid_h = compute_index_and_valid(h_offset, stride_h, in_height, h_in_safe);
                // Loop over kernel width
                for (int kw = 0; kw < kW; kw++) {
                    int w_offset = w + pad_w - kw;
                    int w_in_safe;
                    int valid_w = compute_index_and_valid(w_offset, stride_w, in_width, w_in_safe);

                    // Combine validity flags (will be 1 if valid, 0 otherwise)
                    int valid_mask = valid_d * valid_h * valid_w;

                    // Compute safe input index using the safe coordinates
                    int input_idx = (((n * in_channels + input_channel) * in_depth + d_in_safe) * in_height + h_in_safe) * in_width + w_in_safe;

                    // Compute weight index; weight shape: [in_channels, out_channels_per_group, kT, kH, kW]
                    int weight_idx = ((((input_channel) * out_channels_per_group + out_c_local) * kT + kd) * kH + kh) * kW + kw;

                    // Use the valid mask to conditionally include the contribution.
                    // The ternary operator here should compile to a predicated select, minimizing divergence.
                    scalar_t in_val = valid_mask ? __ldg(&input[input_idx]) : (scalar_t)0;
                    scalar_t weight_val = valid_mask ? weight[weight_idx] : (scalar_t)0;
                    sum += in_val * weight_val;
                }
            }
        }
    }

    // Add bias if provided (this branch is outside inner loops and affects all threads uniformly per channel)
    if (bias != nullptr) {
        sum += __ldg(&bias[c]);
    }

    output[idx] = sum;
}

// Host function to launch the branchless ConvTranspose3d kernel

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Ensure input tensors are contiguous
    input = input.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_tensor;
    if (bias.has_value()) {
        bias_tensor = bias.value().contiguous();
    }

    // Input dimensions
    int N = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    // Kernel dimensions
    int kT = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    // Derive output channels from weight and groups
    int out_channels = weight.size(1) * groups;
    int out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kT + output_padding[0];
    int out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    int out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];

    auto output = torch::zeros({N, out_channels, out_depth, out_height, out_width}, input.options());

    int total_elements = N * out_channels * out_depth * out_height * out_width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_branchless_kernel", ([&] {
        transposed_conv3d_branchless_kernel<scalar_t><<<blocks, threads>>>(
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

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward with branchless control flow to minimize warp divergence",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}
