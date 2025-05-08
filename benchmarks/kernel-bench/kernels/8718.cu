#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Macros for tensor checking
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x);

const int BLOCK_SIZE = 256;

// Declare constant memory for weights (64KB limit on NVIDIA GPUs)
__constant__ float d_weights[16384]; // 64KB / 4 bytes = 16384 floats

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C_in,
    int D_in,
    int H_in,
    int W_in,
    int C_out,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int outD,
    int outH,
    int outW,
    int groups,
    int in_channels_per_group) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C_in * D_in * H_in * W_in;
  if (index >= total) return;
  
  // Decode the flattened index
  int w = index % W_in;
  int tmp = index / W_in;
  int h = tmp % H_in;
  tmp /= H_in;
  int d = tmp % D_in;
  tmp /= D_in;
  int c_in = tmp % C_in;
  int n = tmp / C_in;
  
  int group = c_in / in_channels_per_group;
  float inp = input[index];

  // Cache frequently used values
  int C_out_per_group = C_out / groups;
  int kernel_hw = kernel_h * kernel_w;
  int kernel_dhw = kernel_d * kernel_hw;
  
  #pragma unroll 4
  for (int kd = 0; kd < kernel_d; kd++) {
      int out_d = d * stride_d - pad_d + kd;
      if (out_d >= 0 && out_d < outD) {
          #pragma unroll 4
          for (int kh = 0; kh < kernel_h; kh++) {
              int out_h = h * stride_h - pad_h + kh;
              if (out_h >= 0 && out_h < outH) {
                  #pragma unroll 4
                  for (int kw = 0; kw < kernel_w; kw++) {
                      int out_w = w * stride_w - pad_w + kw;
                      if (out_w >= 0 && out_w < outW) {
                          
                          // Compute base indices once
                          int kernel_base = ((c_in * C_out_per_group * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                          int out_base = (((n * C_out + group * C_out_per_group) * outD + out_d) * outH + out_h) * outW + out_w;
                          
                          #pragma unroll 4
                          for (int oc = 0; oc < C_out_per_group; oc++) {
                              float weight_val = d_weights[kernel_base + oc * kernel_dhw];
                              float out_val = inp * weight_val;
                              atomicAdd(&output[out_base + oc * outD * outH * outW], out_val);
                          }
                      }
                  }
              }
          }
      }
  }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {
  
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  if (bias.has_value()) {
    CHECK_INPUT(*bias);
  }
  
  int N = input.size(0);
  int C_in = input.size(1);
  int D_in = input.size(2);
  int H_in = input.size(3);
  int W_in = input.size(4);
  
  int kernel_d = weight.size(2);
  int kernel_h = weight.size(3);
  int kernel_w = weight.size(4);
  
  int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
  int pad_d = padding[0], pad_h = padding[1], pad_w = padding[2];
  int out_pad_d = output_padding[0], out_pad_h = output_padding[1], out_pad_w = output_padding[2];
  
  int outD = (D_in - 1) * stride_d - 2 * pad_d + kernel_d + out_pad_d;
  int outH = (H_in - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h;
  int outW = (W_in - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w;
  
  int C_out = weight.size(1) * groups;
  
  auto output = torch::zeros({N, C_out, outD, outH, outW}, input.options());
  
  // Copy weights to constant memory
  size_t weight_size = weight.numel() * sizeof(float);
  TORCH_CHECK(weight_size <= 64*1024, "Weight tensor exceeds constant memory limit of 64KB");
  cudaMemcpyToSymbol(d_weights, weight.data_ptr<float>(), weight_size);
  
  const float* input_ptr = input.data_ptr<float>();
  float* output_ptr = output.data_ptr<float>();
  
  int total_input = N * C_in * D_in * H_in * W_in;
  int threads = BLOCK_SIZE;
  int blocks = (total_input + threads - 1) / threads;
  
  int in_channels_per_group = C_in / groups;
  
  conv_transpose3d_kernel<<<blocks, threads>>>(
      input_ptr, output_ptr,
      N, C_in, D_in, H_in, W_in, C_out,
      kernel_d, kernel_h, kernel_w,
      stride_d, stride_h, stride_w,
      pad_d, pad_h, pad_w,
      outD, outH, outW,
      groups, in_channels_per_group);
  
  if (bias.has_value()) {
    output.add_((*bias).view({1, C_out, 1, 1, 1}));
  }
  
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Transposed Conv3D forward (CUDA) with constant memory");
}