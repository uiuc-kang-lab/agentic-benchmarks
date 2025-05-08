#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for transposed 2D convolution with minimized warp divergence
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    // Input dimensions: N, C_in, H_in, W_in
    int N, int C_in, int H_in, int W_in,
    // Output dimensions: C_out, H_out, W_out
    int C_out, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    bool bias_present) {

  // Pre-compute constant values to reduce register pressure
  const int H_W_out = H_out * W_out;
  const int C_H_W_out = C_out * H_W_out;
  const int H_W_in = H_in * W_in;
  const int kernel_hw = kernel_h * kernel_w;
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * C_out * H_out * W_out;
  if (idx >= total) return;

  // Compute output indices more efficiently
  const int w_out = idx % W_out;
  idx /= W_out;
  const int h_out = idx % H_out;
  idx /= H_out;
  const int c_out = idx % C_out;
  const int n = idx / C_out;

  float sum = 0.0f;
  
  #pragma unroll 4
  for (int c = 0; c < C_in; c++) {
    const int input_batch_offset = n * (C_in * H_W_in) + c * H_W_in;
    const int weight_channel_offset = c * (C_out * kernel_hw) + c_out * kernel_hw;
    
    for (int i = 0; i < kernel_h; i++) {
      const int h_diff = h_out + pad_h - i * dilation_h;
      const int h_in_coord = h_diff / stride_h;
      const int mod_h = ((h_diff % stride_h) + stride_h) % stride_h;
      const int mask_h = 1 - int(mod_h != 0);
      
      if (h_in_coord >= 0 && h_in_coord < H_in && mask_h) {
        const int weight_row_offset = weight_channel_offset + i * kernel_w;
        
        for (int j = 0; j < kernel_w; j++) {
          const int w_diff = w_out + pad_w - j * dilation_w;
          const int w_in_coord = w_diff / stride_w;
          const int mod_w = ((w_diff % stride_w) + stride_w) % stride_w;
          const int mask_w = 1 - int(mod_w != 0);
          
          if (w_in_coord >= 0 && w_in_coord < W_in && mask_w) {
            const int input_idx = input_batch_offset + h_in_coord * W_in + w_in_coord;
            const int weight_idx = weight_row_offset + j;
            sum += input[input_idx] * weight[weight_idx];
          }
        }
      }
    }
  }

  // Add bias if provided
  if (bias_present) {
    sum += bias[c_out];
  }

  output[idx] = sum;
}

// Host function that wraps the CUDA kernel launch
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding, // note: output_padding is used in output size calc
    std::vector<int64_t> dilation,
    int64_t groups) {

  // For simplicity, this implementation assumes groups == 1
  auto N = x.size(0);
  auto C_in = x.size(1);
  auto H_in = x.size(2);
  auto W_in = x.size(3);
  auto C_out = weight.size(1);
  int kernel_h = weight.size(2);
  int kernel_w = weight.size(3);

  // Calculate output dimensions using the standard transposed conv formula:
  // H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0]*(kernel_h - 1) + output_padding[0] + 1
  int H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_h - 1) + output_padding[0] + 1;
  int W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_w - 1) + output_padding[1] + 1;

  auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

  int total_threads = N * C_out * H_out * W_out;
  int blockSize = 256;
  int gridSize = (total_threads + blockSize - 1) / blockSize;

  bool bias_present = bias.has_value() && bias.value().defined();
  const float* bias_ptr = bias_present ? bias.value().data_ptr<float>() : nullptr;

  conv_transpose2d_kernel<<<gridSize, blockSize>>>(
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
  m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA) with minimized warp divergence");
}
