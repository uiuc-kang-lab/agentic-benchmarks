#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// This kernel uses a tiling strategy in the spatial dimensions to distribute work evenly across blocks and threads.

template <typename scalar_t>
__global__ void conv_transposed2d_tile_balance_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,  // may be nullptr
    scalar_t* __restrict__ output,

    // Input dimensions
    int N, int C_in, int H_in, int W_in,
    // Output dimensions
    int C_out, int H_out, int W_out,
    // Kernel dimensions
    int kernel_h, int kernel_w,
    // Convolution parameters
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups) {

  // Tiling parameters for spatial dimensions
  constexpr int tile_h = 16;
  constexpr int tile_w = 16;

  // Each block in grid.z corresponds to one (n, c_out) pair
  int n = blockIdx.z / C_out;
  int c_out = blockIdx.z % C_out;

  // Compute the spatial coordinates for this thread in the tile using precomputed base indices
  int h_base = blockIdx.y * tile_h;
  int w_base = blockIdx.x * tile_w;
  int h_out = h_base + threadIdx.y;
  int w_out = w_base + threadIdx.x;

  if (h_out >= H_out || w_out >= W_out) return;

  // Compute output flattened index
  int out_index = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;

  // Determine group boundaries
  int out_channels_per_group = C_out / groups;
  int in_channels_per_group = C_in / groups;
  int group = c_out / out_channels_per_group;

  scalar_t res = 0;
  if (bias != nullptr) {
    res = bias[c_out];
  }

  // For each kernel position, check if this output position aligns with an input pixel
  for (int k_y = 0; k_y < kernel_h; ++k_y) {
    for (int k_x = 0; k_x < kernel_w; ++k_x) {
      int h_offset = h_out + pad_h - k_y * dilation_h;
      int w_offset = w_out + pad_w - k_x * dilation_w;

      if ((h_offset % stride_h) != 0 || (w_offset % stride_w) != 0) continue;

      int h_in = h_offset / stride_h;
      int w_in = w_offset / stride_w;

      if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in) continue;

      // Iterate over input channels in the group
      for (int i = 0; i < in_channels_per_group; i++) {
        int c_in = group * in_channels_per_group + i;
        int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
        // Weight layout: [C_in, out_channels_per_group, kernel_h, kernel_w]
        int weight_idx = (((c_in) * out_channels_per_group + (c_out % out_channels_per_group)) * kernel_h + k_y) * kernel_w + k_x;
        res += input[input_idx] * weight[weight_idx];
      }
    }
  }

  output[out_index] = res;
}


torch::Tensor conv_transposed2d_cuda_tile_balance(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,         // [stride_h, stride_w]
    std::vector<int64_t> padding,        // [pad_h, pad_w]
    std::vector<int64_t> output_padding, // [output_pad_h, output_pad_w]
    std::vector<int64_t> dilation,       // [dilation_h, dilation_w]
    int64_t groups) {

  // Input dimensions: [N, C_in, H_in, W_in]
  int N = input.size(0);
  int C_in = input.size(1);
  int H_in = input.size(2);
  int W_in = input.size(3);

  int kernel_h = weight.size(2);
  int kernel_w = weight.size(3);

  int stride_h = stride[0];
  int stride_w = stride[1];
  int pad_h = padding[0];
  int pad_w = padding[1];
  int output_pad_h = output_padding[0];
  int output_pad_w = output_padding[1];
  int dilation_h = dilation[0];
  int dilation_w = dilation[1];

  // Weight shape: [C_in, out_channels_per_group, kernel_h, kernel_w]
  int out_channels_per_group = weight.size(1);
  int C_out = out_channels_per_group * groups;

  // Compute output spatial dimensions using the transposed convolution formula
  int H_out = (H_in - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + output_pad_h + 1;
  int W_out = (W_in - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + output_pad_w + 1;

  auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

  // Tiling parameters
  constexpr int tile_h = 16;
  constexpr int tile_w = 16;

  // Set up a 3D grid where grid.z covers (n, c_out)
  dim3 block(tile_w, tile_h);
  dim3 grid((W_out + tile_w - 1) / tile_w,
            (H_out + tile_h - 1) / tile_h,
            N * C_out);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transposed2d_cuda_tile_balance", ([&] {
    conv_transposed2d_tile_balance_kernel<scalar_t><<<grid, block>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr,
        output.data_ptr<scalar_t>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        groups);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transposed2d_cuda_tile_balance, "Balanced ConvTranspose2D forward (CUDA)");
}
