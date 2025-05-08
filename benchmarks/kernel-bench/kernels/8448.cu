#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void aligned_conv_transpose2d_kernel(
    const float* const __restrict__ input,
    const float* const __restrict__ weight,
    const float* const __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const bool bias_present) {

  __align__(16) float sum = 0.0f;
  
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * C_out * H_out * W_out;
  if (idx >= total) return;

  const int w_out = idx % W_out;
  const int tmp = idx / W_out;
  const int h_out = tmp % H_out;
  const int rem = tmp / H_out;
  const int c_out = rem % C_out;
  const int n = rem / C_out;

  const int input_batch_offset = n * (C_in * H_in * W_in);
  const int weight_cout_offset = c_out * (kernel_h * kernel_w);
  
  #pragma unroll 4
  for (int c = 0; c < C_in; c++) {
    const int input_channel_offset = input_batch_offset + c * (H_in * W_in);
    const int weight_channel_offset = c * (C_out * kernel_h * kernel_w) + weight_cout_offset;
    
    #pragma unroll 2
    for (int i = 0; i < kernel_h; i++) {
      const int h_diff = h_out + pad_h - i * dilation_h;
      const int h_in_coord = h_diff / stride_h;
      const bool h_valid = (h_diff % stride_h == 0) && (h_in_coord >= 0) && (h_in_coord < H_in);
      
      if (h_valid) {
        const int input_h_offset = input_channel_offset + h_in_coord * W_in;
        const int weight_h_offset = weight_channel_offset + i * kernel_w;
        
        #pragma unroll 4
        for (int j = 0; j < kernel_w; j++) {
          const int w_diff = w_out + pad_w - j * dilation_w;
          const int w_in_coord = w_diff / stride_w;
          
          if ((w_diff % stride_w == 0) && (w_in_coord >= 0) && (w_in_coord < W_in)) {
            const float input_val = __ldg(&input[input_h_offset + w_in_coord]);
            const float weight_val = __ldg(&weight[weight_h_offset + j]);
            sum = __fmaf_rn(input_val, weight_val, sum);
          }
        }
      }
    }
  }

  if (bias_present) {
    sum += __ldg(&bias[c_out]);
  }

  output[idx] = sum;
}

torch::Tensor aligned_conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

  const auto N = x.size(0);
  const auto C_in = x.size(1);
  const auto H_in = x.size(2);
  const auto W_in = x.size(3);
  const auto C_out = weight.size(1);
  const int kernel_h = weight.size(2);
  const int kernel_w = weight.size(3);

  const int H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_h - 1) + output_padding[0] + 1;
  const int W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_w - 1) + output_padding[1] + 1;

  auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

  const int total_threads = N * C_out * H_out * W_out;
  const int blockSize = 256;
  const int gridSize = (total_threads + blockSize - 1) / blockSize;

  const bool bias_present = bias.has_value() && bias.value().defined();
  const float* bias_ptr = bias_present ? bias.value().data_ptr<float>() : nullptr;

  aligned_conv_transpose2d_kernel<<<gridSize, blockSize>>>(
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
  m.def("forward", &aligned_conv_transpose2d_cuda, "Aligned memory ConvTranspose2D forward (CUDA)");
}