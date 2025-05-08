#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <int UNROLL_FACTOR>
__global__ void conv_transpose2d_kernel(
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

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * C_out * H_out * W_out) return;

  // Output tensor coordinates
  int w_out = idx % W_out;
  int h_out = (idx / W_out) % H_out;
  int c_out = (idx / (W_out * H_out)) % C_out;
  int n = idx / (W_out * H_out * C_out);

  float sum = 0.0f;
  
  for (int c = 0; c < C_in; c++) {
    #pragma unroll
    for (int i = 0; i < kernel_h; i++) {
      #pragma unroll
      for (int j = 0; j < kernel_w; j += UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
          if (j + u >= kernel_w) break;
          
          // Input position calculation
          int h_in = h_out + pad_h - i * dilation_h;
          int w_in = w_out + pad_w - (j + u) * dilation_w;

          // Branchless validity checks
          int h_valid = (h_in % stride_h == 0) && (h_in / stride_h < H_in) && (h_in / stride_h >= 0);
          int w_valid = (w_in % stride_w == 0) && (w_in / stride_w < W_in) && (w_in / stride_w >= 0);
          
          if (h_valid && w_valid) {
            int h_coord = h_in / stride_h;
            int w_coord = w_in / stride_w;
            
            float input_val = input[n * C_in * H_in * W_in + c * H_in * W_in + h_coord * W_in + w_coord];
            float weight_val = weight[c * C_out * kernel_h * kernel_w + c_out * kernel_h * kernel_w + i * kernel_w + j + u];
            sum += input_val * weight_val;
          }
        }
      }
    }
  }

  if (bias_present) sum += bias[c_out];
  output[idx] = sum;
}

torch::Tensor conv_transpose2d_cuda(
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

  int total = N * C_out * H_out * W_out;
  // Use 2D thread blocks for better memory access patterns
  dim3 block(16, 8);  // 128 threads per block, still multiple of warp size
  dim3 grid(
      (W_out + block.x - 1) / block.x,
      (H_out + block.y - 1) / block.y,
      N * C_out  // Batch and output channels in z dimension
  );

  // Dynamic unroll factor selection based on kernel width
  if (kernel_w % 4 == 0) {
    conv_transpose2d_kernel<4><<<grid, block>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kernel_h, kernel_w,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        bias.has_value());
  } else {
    conv_transpose2d_kernel<2><<<grid, block>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kernel_h, kernel_w,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        bias.has_value());
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_cuda, "Optimized ConvTranspose2D with dynamic unrolling");
}