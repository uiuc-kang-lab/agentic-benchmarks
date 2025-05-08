#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized kernel using __ldg for read-only global memory loads and assuming data is 128-bit aligned

template <typename scalar_t>
__global__ void conv_transposed2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,  // can be nullptr if not provided
    scalar_t* output,

    // Input dimensions
    const int N,
    const int in_channels,
    const int in_height,
    const int in_width,

    // Output dimensions
    const int out_channels,
    const int out_height,
    const int out_width,

    // Kernel dimensions
    const int kernel_h,
    const int kernel_w,

    // Convolution parameters
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups) {

  // Each thread computes one output element
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * out_channels * out_height * out_width;
  if (index >= total) return;

  // Compute n, c_out, h, w from the flattened index
  int w_out = index % out_width;
  int tmp = index / out_width;
  int h_out = tmp % out_height;
  tmp = tmp / out_height;
  int c_out = tmp % out_channels;
  int n = tmp / out_channels;

  // Determine channel group information
  int out_channels_per_group = out_channels / groups;
  int in_channels_per_group = in_channels / groups;
  int group = c_out / out_channels_per_group;

  // Initialize accumulation with bias if provided
  scalar_t sum = 0;
  if (bias != nullptr) {
    sum = __ldg(&bias[c_out]);
  }

  // For transposed convolution, each output pixel (n, c_out, h_out, w_out) receives contributions
  // from input pixels whose location satisfy:
  //   h_out + pad_h - k_h * dilation_h = in_y * stride_h
  //   w_out + pad_w - k_w * dilation_w = in_x * stride_w
  // for some kernel offsets (k_h, k_w) and valid in_y, in_x
  for (int k_h = 0; k_h < kernel_h; k_h++) {
    for (int k_w = 0; k_w < kernel_w; k_w++) {
      int tmp_y = h_out + pad_h - k_h * dilation_h;
      int tmp_x = w_out + pad_w - k_w * dilation_w;
      // Check if the coordinate maps exactly to an input pixel
      if ((tmp_y % stride_h == 0) && (tmp_x % stride_w == 0)) {
        int in_y = tmp_y / stride_h;
        int in_x = tmp_x / stride_w;
        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
          // For the appropriate group, sum over the corresponding input channels
          for (int c = 0; c < in_channels_per_group; c++) {
            int input_channel = group * in_channels_per_group + c;
            int input_idx = ((n * in_channels + input_channel) * in_height + in_y) * in_width + in_x;
            // Use __ldg for read-only load from input. The pointer is assumed to be aligned to 128-bit boundaries.
            scalar_t input_val = __ldg(&input[input_idx]);
            // Weight layout: [in_channels, out_channels_per_group, kernel_h, kernel_w]
            // Mapping: for a given input channel, the output channel within the group
            int weight_idx = (((input_channel) * out_channels_per_group + (c_out % out_channels_per_group)) * kernel_h + k_h) * kernel_w + k_w;
            scalar_t weight_val = __ldg(&weight[weight_idx]);
            sum += input_val * weight_val;
          }
        }
      }
    }
  }

  int output_idx = ((n * out_channels + c_out) * out_height + h_out) * out_width + w_out;
  output[output_idx] = sum;
}


// Host function to prepare kernel launch
// NOTE: This implementation computes the output dimensions based on the standard conv_transpose2d formula:
// out_dim = (in_dim - 1) * stride - 2 * pad + dilation * (kernel_size - 1) + output_padding + 1

torch::Tensor conv_transpose2d_cuda_optimized(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

  // Input tensor dimensions: [N, in_channels, in_height, in_width]
  auto N = input.size(0);
  auto in_channels = input.size(1);
  auto in_height = input.size(2);
  auto in_width = input.size(3);

  // Weight tensor dimensions: [in_channels, out_channels_per_group, kernel_h, kernel_w]
  auto kernel_h = weight.size(2);
  auto kernel_w = weight.size(3);
  int out_channels_per_group = weight.size(1);
  int out_channels = out_channels_per_group * groups;

  // Compute output dimensions
  int out_height = (in_height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_h - 1) + output_padding[0] + 1;
  int out_width = (in_width - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_w - 1) + output_padding[1] + 1;

  auto output = torch::zeros({N, out_channels, out_height, out_width}, input.options());

  const int threads = 256;
  const int total = N * out_channels * out_height * out_width;
  const int blocks = (total + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda_optimized", ([&] {
    conv_transposed2d_kernel<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr,
        output.data_ptr<scalar_t>(),
        N,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_h,
        kernel_w,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        groups);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_cuda_optimized, "ConvTranspose2D forward optimized (CUDA)");
}
