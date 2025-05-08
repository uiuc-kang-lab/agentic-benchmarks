#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel with optimized block and thread indexing

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
  // Define a 3D indexing scheme for block and thread
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int d = blockIdx.z * blockDim.z + threadIdx.z;

  if (w >= out_width || h >= out_height || d >= out_depth) return;

  // Loop over batch size and out_channels to further utilize concurrency
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < out_channels; c++) {
      // Determine group and local channel index
      int group = c / (out_channels / groups);
      int out_c_local = c % (out_channels / groups);

      scalar_t sum = 0;

      // Number of input channels per group
      int in_channels_per_group = in_channels / groups;

      // Loop over the input channels and compute the convolution
      for (int ic = 0; ic < in_channels_per_group; ic++) {
        int input_channel = group * in_channels_per_group + ic;
        for (int kd = 0; kd < kT; kd++) {
          int d_in_tmp = d + pad_d - kd;
          if (d_in_tmp % stride_d != 0) continue;
          int d_in = d_in_tmp / stride_d;
          if (d_in < 0 || d_in >= in_depth) continue;

          for (int kh = 0; kh < kH; kh++) {
            int h_in_tmp = h + pad_h - kh;
            if (h_in_tmp % stride_h != 0) continue;
            int h_in = h_in_tmp / stride_h;
            if (h_in < 0 || h_in >= in_height) continue;

            for (int kw = 0; kw < kW; kw++) {
              int w_in_tmp = w + pad_w - kw;
              if (w_in_tmp % stride_w != 0) continue;
              int w_in = w_in_tmp / stride_w;
              if (w_in < 0 || w_in >= in_width) continue;

              // Compute flat index for input: N x in_channels x in_depth x in_height x in_width
              int input_idx = (((n * in_channels + input_channel) * in_depth + d_in) * in_height + h_in) * in_width + w_in;

              // Compute flat index for weight:
              // Weight shape: [in_channels, out_channels/groups, kT, kH, kW]
              int weight_idx = ((((input_channel) * (out_channels / groups) + out_c_local) * kT + kd) * kH + kh) * kW + kw;

              sum += input[input_idx] * weight[weight_idx];
            }
          }
        }
      }

      if (bias != nullptr) {
        sum += bias[c];
      }

      // Compute flat index for output
      int output_idx = (((n * out_channels + c) * out_depth + d) * out_height + h) * out_width + w;

      output[output_idx] = sum;
    }
  }
}

// Host function that sets up optimised grid dimensions, launches kernel with optimized block and thread indexing

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Ensure tensors are contiguous
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

    // out_channels: weight has shape [in_channels, out_channels/groups, kT, kH, kW]
    int out_channels = weight.size(1) * groups;

    // Compute output dimensions using the transposed convolution formula:
    // out_dim = (in_dim - 1) * stride - 2 * padding + kernel_size + output_padding
    int out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kT + output_padding[0];
    int out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    int out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];

    // Allocate output tensor
    auto output = torch::zeros({N, out_channels, out_depth, out_height, out_width}, input.options());

    // Choose block and grid dimensions that optimize 3D parallelism
    dim3 threads(8, 8, 4);  // Use a 3D block
    dim3 blocks((out_width + threads.x - 1) / threads.x,
                (out_height + threads.y - 1) / threads.y,
                (out_depth + threads.z - 1) / threads.z);

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

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized ConvTranspose3d forward function with improved thread/block indexing",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}