#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x);


// Experimentally, we found a block size of 256 to be optimal on the NVIDIA H100 GPU
const int BLOCK_SIZE = 256;

// This kernel performs the 3D transposed convolution operation.
// Each thread processes one input element and iterates over the kernel dimensions and output channels
// to accumulate its contribution in the output tensor using atomicAdd.
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
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
  
  // Decode the flattened index into (n, c_in, d, h, w) coordinates
  int w = index % W_in;
  int tmp = index / W_in;
  int h = tmp % H_in;
  tmp /= H_in;
  int d = tmp % D_in;
  tmp /= D_in;
  int c_in = tmp % C_in;
  tmp /= C_in;
  int n = tmp;
  
  // Determine the group for the current input channel
  int group = c_in / in_channels_per_group;

  // Get the input value
  float inp = input[index];
  
  // Iterate over each kernel element
  for (int kd = 0; kd < kernel_d; kd++) {
      int out_d = d * stride_d - pad_d + kd;
      if (out_d < 0 || out_d >= outD) continue;
      for (int kh = 0; kh < kernel_h; kh++) {
          int out_h = h * stride_h - pad_h + kh;
          if (out_h < 0 || out_h >= outH) continue;
          for (int kw = 0; kw < kernel_w; kw++) {
              int out_w = w * stride_w - pad_w + kw;
              if (out_w < 0 || out_w >= outW) continue;
              
              // For the given input channel, iterate over the corresponding output channels
              for (int oc = 0; oc < (C_out / groups); oc++) {
                  // Compute the weight index; weight shape is [C_in, C_out/groups, kernel_d, kernel_h, kernel_w]
                  int weight_index = (((c_in * (C_out / groups) + oc) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                  float weight_val = weight[weight_index];
                  float out_val = inp * weight_val;
                  
                  int oc_global = group * (C_out / groups) + oc;
                  int output_index = (((n * C_out + oc_global) * outD + out_d) * outH + out_h) * outW + out_w;

                  atomicAdd(&output[output_index], out_val);
              }
          }
      }
  }
}

// Kernel to add bias to the output tensor. The bias is applied per output channel.
__global__ void add_bias_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int total,
    int C_out,
    int outD,
    int outH,
    int outW) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    
    // Decode into tensor coordinates [N, C_out, outD, outH, outW]
    int w = index % outW;
    int tmp = index / outW;
    int h = tmp % outH;
    tmp /= outH;
    int d = tmp % outD;
    tmp /= outD;
    int c = tmp % C_out;
    // n is not needed as bias is per-channel and broadcasted over batch and spatial dims
    
    output[index] += bias[c];
}

// Host function implementing the forward pass of the transposed 3D convolution
// This function computes the output dimensions based on stride, padding, and output_padding,
// then launches the CUDA kernels with the tuned block configuration.

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
  
  // Input dimensions: [N, C_in, D_in, H_in, W_in]
  int N = input.size(0);
  int C_in = input.size(1);
  int D_in = input.size(2);
  int H_in = input.size(3);
  int W_in = input.size(4);
  
  // Kernel dimensions from weight: [C_in, C_out/groups, kernel_d, kernel_h, kernel_w]
  int kernel_d = weight.size(2);
  int kernel_h = weight.size(3);
  int kernel_w = weight.size(4);
  
  int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
  int pad_d = padding[0], pad_h = padding[1], pad_w = padding[2];
  int out_pad_d = output_padding[0], out_pad_h = output_padding[1], out_pad_w = output_padding[2];
  
  // Calculate output dimensions using the transposed convolution formula
  int outD = (D_in - 1) * stride_d - 2 * pad_d + kernel_d + out_pad_d;
  int outH = (H_in - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h;
  int outW = (W_in - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w;
  
  // Compute the number of output channels
  int C_out = weight.size(1) * groups;
  
  // Allocate and zero-initialize the output tensor (shape: [N, C_out, outD, outH, outW])
  auto output = torch::zeros({N, C_out, outD, outH, outW}, input.options());
  
  // Get raw pointers
  const float* input_ptr = input.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  float* output_ptr = output.data_ptr<float>();
  
  int total_input = N * C_in * D_in * H_in * W_in;
  int threads = BLOCK_SIZE;
  int blocks = (total_input + threads - 1) / threads;

  int in_channels_per_group = C_in / groups;

  // Launch the main convolution kernel with the tuned block size
  conv_transpose3d_kernel<<<blocks, threads>>>(input_ptr, weight_ptr, output_ptr,
                                               N, C_in, D_in, H_in, W_in, C_out,
                                               kernel_d, kernel_h, kernel_w,
                                               stride_d, stride_h, stride_w,
                                               pad_d, pad_h, pad_w,
                                               outD, outH, outW,
                                               groups, in_channels_per_group);

  // If a bias is provided, launch a secondary kernel to add it
  if (bias.has_value()) {
    const float* bias_ptr = (*bias).data_ptr<float>();
    int total_output = N * C_out * outD * outH * outW;
    int threads_bias = BLOCK_SIZE;
    int blocks_bias = (total_output + threads_bias - 1) / threads_bias;
    add_bias_kernel<<<blocks_bias, threads_bias>>>(output_ptr, bias_ptr, total_output, C_out, outD, outH, outW);
  }
  
  cudaDeviceSynchronize();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Transposed Conv3D forward (CUDA) with tuned block size");
}
