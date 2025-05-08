#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x);

// Tuned block size
const int BLOCK_SIZE = 256;

// Device helper: decode a flat index into 5D coordinates [N, C, D, H, W]
__device__ inline void decode_input_index(int index, int W, int H, int D, int C, int &n, int &c, int &d, int &h, int &w) {
    // Order: (n, c, d, h, w) with w fastest and n slowest
    w = index % W;
    int tmp = index / W;
    h = tmp % H;
    tmp /= H;
    d = tmp % D;
    tmp /= D;
    c = tmp % C;
    n = tmp / C;
}

// Device helper: calculate flat output index for coordinates [n, oc, d, h, w]
__device__ inline int calc_output_index(int n, int C_out, int outD, int outH, int outW, int oc, int d, int h, int w) {
    return (((n * C_out + oc) * outD + d) * outH + h) * outW + w;
}

// Device helper: calculate flat weight index
__device__ inline int calc_weight_index(int c_in, int oc, int kd, int kh, int kw,
                                          int kernel_d, int kernel_h, int kernel_w, int out_channels_per_group) {
    return (((c_in * out_channels_per_group + oc) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
}

// Device helper: decode the output channel from a flat output index (for bias kernel)
__device__ inline int decode_output_channel(int index, int outW, int outH, int outD, int C_out) {
    // Given output tensor of shape [N, C_out, outD, outH, outW]
    int tmp = index / outW;  // remove w
    tmp /= outH;             // remove h
    tmp /= outD;             // remove d
    return tmp % C_out;      // remaining is channel
}

// 3D Transposed Convolution Kernel with grid-stride loop
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int outD, int outH, int outW,
    int groups, int in_channels_per_group) {
  int total = N * C_in * D_in * H_in * W_in;

  // Grid-stride loop
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += blockDim.x * gridDim.x) {
    int n, c_in, d, h, w;
    decode_input_index(index, W_in, H_in, D_in, C_in, n, c_in, d, h, w);

    int group = c_in / in_channels_per_group;
    float inp_val = input[index];

    // Loop over kernel dimensions
    for (int kd = 0; kd < kernel_d; kd++) {
      int out_d = d * stride_d - pad_d + kd;
      if (out_d < 0 || out_d >= outD) continue;
      for (int kh = 0; kh < kernel_h; kh++) {
        int out_h = h * stride_h - pad_h + kh;
        if (out_h < 0 || out_h >= outH) continue;
        for (int kw = 0; kw < kernel_w; kw++) {
          int out_w = w * stride_w - pad_w + kw;
          if (out_w < 0 || out_w >= outW) continue;
          
          // Loop over output channels within the group
          for (int oc = 0; oc < (C_out / groups); oc++) {
            int weight_index = calc_weight_index(c_in, oc, kd, kh, kw, kernel_d, kernel_h, kernel_w, C_out / groups);
            float weight_val = weight[weight_index];
            float out_val = inp_val * weight_val;
            int oc_global = group * (C_out / groups) + oc;
            int o_index = calc_output_index(n, C_out, outD, outH, outW, oc_global, out_d, out_h, out_w);
            atomicAdd(&output[o_index], out_val);
          }
        }
      }
    }
  }
}

// Kernel to add bias using grid-stride loops
__global__ void add_bias_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int total,
    int C_out,
    int outD, int outH, int outW) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += blockDim.x * gridDim.x) {
    int c = decode_output_channel(index, outW, outH, outD, C_out);
    output[index] += bias[c];
  }
}

// Host function implementing the forward pass
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

  // Weight dimensions: [C_in, C_out/groups, kernel_d, kernel_h, kernel_w]
  int kernel_d = weight.size(2);
  int kernel_h = weight.size(3);
  int kernel_w = weight.size(4);

  int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
  int pad_d = padding[0], pad_h = padding[1], pad_w = padding[2];
  int out_pad_d = output_padding[0], out_pad_h = output_padding[1], out_pad_w = output_padding[2];

  // Calculate output dimensions
  int outD = (D_in - 1) * stride_d - 2 * pad_d + kernel_d + out_pad_d;
  int outH = (H_in - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h;
  int outW = (W_in - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w;

  // Total number of output channels
  int C_out = weight.size(1) * groups;

  // Allocate output tensor and zero-initialize
  auto output = torch::zeros({N, C_out, outD, outH, outW}, input.options());

  int total_input = N * C_in * D_in * H_in * W_in;
  int conv_blocks = (total_input + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int in_channels_per_group = C_in / groups;

  // Create CUDA streams for concurrent execution
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  // Launch convolution kernel on stream1
  conv_transpose3d_kernel<<<conv_blocks, BLOCK_SIZE, 0, stream1>>>(
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
      output.data_ptr<float>(),
      N, C_in, D_in, H_in, W_in,
      C_out,
      kernel_d, kernel_h, kernel_w,
      stride_d, stride_h, stride_w,
      pad_d, pad_h, pad_w,
      outD, outH, outW,
      groups, in_channels_per_group);

  // Launch bias addition kernel if bias is provided on stream2
  if (bias.has_value()) {
    int total_output = N * C_out * outD * outH * outW;
    int bias_blocks = (total_output + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_bias_kernel<<<bias_blocks, BLOCK_SIZE, 0, stream2>>>(
        output.data_ptr<float>(),
        (*bias).data_ptr<float>(),
        total_output,
        C_out, outD, outH, outW);
  }

  // Synchronize both streams
  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);
  
  // Cleanup streams
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Modular Transposed Conv3D forward (CUDA) with device functions");
}
