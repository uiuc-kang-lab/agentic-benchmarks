#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Macros for tensor checking
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x);


// Define warp and block sizes
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;  // Each block has 256 threads (8 warps per block)

// Warp-level reduction using shuffle instructions
__inline__ __device__ float warp_reduce_sum(float val) {
  // All 32 threads in a warp participate
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// This kernel implements a gather-based transposed 3D convolution.
// Each warp computes one output element by reducing contributions across the reduction domain
// (input channels within a group and kernel volume). The reduction is split among warp lanes
// and combined using warp-level primitives, removing the need for expensive global atomics.
__global__ void gather_conv_transpose3d_kernel(
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

  // Each warp computes one output element
  int warpsPerBlock = blockDim.x / WARP_SIZE;
  int global_warp_idx = blockIdx.x * warpsPerBlock + (threadIdx.x / WARP_SIZE);
  int lane = threadIdx.x % WARP_SIZE;

  // Total number of output elements
  int total_outputs = N * C_out * outD * outH * outW;
  if (global_warp_idx >= total_outputs) return;

  // Decode the linear output index into (n, oc, od, oh, ow) coordinates
  int tmp = global_warp_idx;
  int ow = tmp % outW; tmp /= outW;
  int oh = tmp % outH; tmp /= outH;
  int od = tmp % outD; tmp /= outD;
  int oc = tmp % C_out;
  int n = tmp / C_out;  // remaining is the batch index

  // Determine group for the output channel
  int C_out_per_group = C_out / groups;
  int group = oc / C_out_per_group;
  int oc_in_group = oc - group * C_out_per_group;

  // The reduction domain covers all input channels in this group and the kernel volume
  int reduction_size = in_channels_per_group * kernel_d * kernel_h * kernel_w;
  float sum = 0.0f;

  // Each warp lane accumulates over a strided portion of the reduction domain
  for (int r = lane; r < reduction_size; r += WARP_SIZE) {
    int c_offset = r / (kernel_d * kernel_h * kernel_w);
    int rem = r % (kernel_d * kernel_h * kernel_w);
    int kd = rem / (kernel_h * kernel_w);
    int rem2 = rem % (kernel_h * kernel_w);
    int kh = rem2 / kernel_w;
    int kw = rem2 % kernel_w;

    int c = group * in_channels_per_group + c_offset;  // Actual input channel index

    // Compute the corresponding input spatial coordinates from the output coordinate
    int d_in_val = od + pad_d - kd;
    int h_in_val = oh + pad_h - kh;
    int w_in_val = ow + pad_w - kw;

    // Check if the location aligns with the stride and is within input bounds
    if ( (d_in_val % stride_d == 0) && (h_in_val % stride_h == 0) && (w_in_val % stride_w == 0) ) {
      int d_in = d_in_val / stride_d;
      int h_in = h_in_val / stride_h;
      int w_in = w_in_val / stride_w;
      if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
        // Compute linear index for input: [n, c, d_in, h_in, w_in]
        int input_index = (((n * C_in + c) * D_in + d_in) * H_in + h_in) * W_in + w_in;
        // Compute linear index for weight: [c, oc_in_group, kd, kh, kw]
        int weight_index = ((((c) * C_out_per_group + oc_in_group) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
        sum += input[input_index] * weight[weight_index];
      }
    }
  }

  // Perform warp-level reduction to sum contributions from all lanes
  sum = warp_reduce_sum(sum);

  // Lane 0 writes the final output
  if (lane == 0) {
    output[global_warp_idx] = sum;
  }
}

// Kernel to add bias to the output tensor. Bias is applied per output channel
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
  int w = index % outW;
  int tmp = index / outW;
  int h = tmp % outH;
  tmp /= outH;
  int d = tmp % outD;
  tmp /= outD;
  int c = tmp % C_out;
  output[index] += bias[c];
}

// Host function for the forward pass
// Computes output dimensions and launches the gather-based kernel with warp-level reduction
// Note: This implementation avoids global atomics by having each warp compute one output element
// via intra-warp reduction using __shfl_down_sync().

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

  // Weight dimensions: [C_in, C_out_per_group, kernel_d, kernel_h, kernel_w]
  int kernel_d = weight.size(2);
  int kernel_h = weight.size(3);
  int kernel_w = weight.size(4);

  int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
  int pad_d = padding[0], pad_h = padding[1], pad_w = padding[2];
  int out_pad_d = output_padding[0], out_pad_h = output_padding[1], out_pad_w = output_padding[2];

  // Compute output dimensions using the transposed convolution formula
  int outD = (D_in - 1) * stride_d - 2 * pad_d + kernel_d + out_pad_d;
  int outH = (H_in - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h;
  int outW = (W_in - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w;

  // Total number of output channels
  int C_out_per_group = weight.size(1);
  int C_out = C_out_per_group * groups;

  // Allocate the output tensor (shape: [N, C_out, outD, outH, outW])
  auto output = torch::empty({N, C_out, outD, outH, outW}, input.options());

  // Launch the gather-based kernel
  int total_output = N * C_out * outD * outH * outW;
  // Each warp (of 32 threads) computes one output element
  int warpsPerBlock = BLOCK_SIZE / WARP_SIZE;
  int numBlocks = (total_output + warpsPerBlock - 1) / warpsPerBlock;

  const float* input_ptr = input.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  float* output_ptr = output.data_ptr<float>();

  int in_channels_per_group = C_in / groups;
  gather_conv_transpose3d_kernel<<<numBlocks, BLOCK_SIZE>>>(
      input_ptr,
      weight_ptr,
      output_ptr,
      N, C_in, D_in, H_in, W_in,
      C_out,
      kernel_d, kernel_h, kernel_w,
      stride_d, stride_h, stride_w,
      pad_d, pad_h, pad_w,
      outD, outH, outW,
      groups, in_channels_per_group);

  // If a bias is provided, add it using a separate kernel
  if (bias.has_value()) {
    const float* bias_ptr = (*bias).data_ptr<float>();
    int total_elements = N * C_out * outD * outH * outW;
    int threads_bias = BLOCK_SIZE;
    int blocks_bias = (total_elements + threads_bias - 1) / threads_bias;
    add_bias_kernel<<<blocks_bias, threads_bias>>>(output_ptr, bias_ptr, total_elements, C_out, outD, outH, outW);
  }

  cudaDeviceSynchronize();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Transposed Conv3D forward (CUDA) with warp-level reduction");
}
