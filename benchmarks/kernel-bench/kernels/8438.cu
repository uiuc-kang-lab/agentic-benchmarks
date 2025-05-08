#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized CUDA kernel for transposed 2D convolution
__global__ void optimized_conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    bool bias_present) {

  // Use a 3D grid: blockIdx.x for width, blockIdx.y for height, and blockIdx.z for (n * C_out)
  int w_out = blockIdx.x * blockDim.x + threadIdx.x;
  int h_out = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.z / C_out;
  int c_out = blockIdx.z % C_out;
  if (w_out >= W_out || h_out >= H_out) return;

  float sum = 0.0f;
  
  for (int c = 0; c < C_in; c++) {
    for (int i = 0; i < kernel_h; i++) {
      for (int j = 0; j < kernel_w; j++) {
        int h_diff = h_out + pad_h - i * dilation_h;
        int w_diff = w_out + pad_w - j * dilation_w;

        int mod_h = ((h_diff % stride_h) + stride_h) % stride_h;
        int mod_w = ((w_diff % stride_w) + stride_w) % stride_w;
        int mask_h = 1 - int(mod_h != 0);
        int mask_w = 1 - int(mod_w != 0);

        int valid_mask = mask_h * mask_w;

        int h_in_coord = h_diff / stride_h;
        int w_in_coord = w_diff / stride_w;

        int in_bound = (h_in_coord >= 0 && h_in_coord < H_in && w_in_coord >= 0 && w_in_coord < W_in) ? 1 : 0;
        valid_mask *= in_bound;

        if (valid_mask) {
          int input_index = n * (C_in * H_in * W_in) + c * (H_in * W_in) + h_in_coord * W_in + w_in_coord;
          int weight_index = c * (C_out * kernel_h * kernel_w) + c_out * (kernel_h * kernel_w) + i * kernel_w + j;
          float input_val = input[input_index];
          float weight_val = weight[weight_index];
          sum += input_val * weight_val;
        }
      }
    }
  }

  if (bias_present) {
    sum += bias[c_out];
  }

  output[idx] = sum;
}

// Host function that wraps the CUDA kernel launch
torch::Tensor optimized_conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

  auto N = x.size(0);
  auto C_in = x.size(1);
  auto H_in = x.size(2);
  auto W_in = x.size(3);
  auto C_out = weight.size(1);
  int kernel_h = weight.size(2);
  int kernel_w = weight.size(3);

  int H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_h - 1) + output_padding[0] + 1;
  int W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_w - 1) + output_padding[1] + 1;

  auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

  int total_threads = N * C_out * H_out * W_out;
  int blockSize = 256;
  int gridSize = (total_threads + blockSize - 1) / blockSize;

  bool bias_present = bias.has_value() && bias.value().defined();
  const float* bias_ptr = bias_present ? bias.value().data_ptr<float>() : nullptr;

  optimized_conv_transpose2d_kernel<<<gridSize, blockSize>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias_ptr,
      output.data_ptr<float>(),
      N, C_in, H_in, W_in,
      C_out, H_out, W_out,
      kernel_h, kernel_w,
      stride[0], stride[1],
      padding[0], padding[1],
      dilation[0], dilation[1],
      bias_present);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &optimized_conv_transpose2d_cuda, "Optimized ConvTranspose2D forward (CUDA)");
}
