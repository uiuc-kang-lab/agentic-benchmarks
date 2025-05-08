#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>
#include <stdio.h>

// Compute greatest common divisor
__device__ int gcd(int a, int b) {
  while (b != 0) {
    int t = b;
    b = a % b;
    a = t;
  }
  return a;
}

// Compute minimum of two integers
__device__ int my_min(int a, int b) {
  return a < b ? a : b;
}

// CUDA kernel for 2D transposed convolution with manual loop unrolling for the inner channel loop
__global__ void conv_transpose2d_kernel_manual_unroll(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int in_channels_per_group,
    const int out_channels_per_group) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * out_channels * out_h * out_w;
  if (index >= total) return;

  // Decode flat index into (n, oc, oh, ow)
  int ow = index % out_w;
  int tmp = index / out_w;
  int oh = tmp % out_h;
  tmp = tmp / out_h;
  int oc = tmp % out_channels;
  int n = tmp / out_channels;

  // Initialize output value with bias
  float out_val = bias[oc];

  // Determine group index
  int g = oc / out_channels_per_group;

  // Precompute candidate positions (output coordinate plus padding)
  int candidate_h = oh + pad_h;
  int candidate_w = ow + pad_w;

  // Compute valid kernel offsets and steps for height dimension
  int offset_kh = -1;
  int mod_h = candidate_h % stride_h;
  for (int k = 0; k < stride_h; k++) {
    if ((k * dilation_h) % stride_h == mod_h) {
      offset_kh = k;
      break;
    }
  }
  int step_kh = stride_h / gcd(stride_h, dilation_h);
  int kh_bound = candidate_h / dilation_h + 1;
  int kh_end = my_min(kernel_h, kh_bound);

  // Compute valid kernel offsets and steps for width dimension
  int offset_kw = -1;
  int mod_w = candidate_w % stride_w;
  for (int k = 0; k < stride_w; k++) {
    if ((k * dilation_w) % stride_w == mod_w) {
      offset_kw = k;
      break;
    }
  }
  int step_kw = stride_w / gcd(stride_w, dilation_w);
  int kw_bound = candidate_w / dilation_w + 1;
  int kw_end = my_min(kernel_w, kw_bound);

  // Pre-calculate constants for input indexing
  int stride_c = in_h * in_w; // offset between consecutive channels in input
  int base_n = n * in_channels * stride_c;  // base offset for the nth example

  // Loop over kernel height and width with unrolling
  #pragma unroll
  for (int kh = offset_kh; kh < kh_end; kh += step_kh) {
    int h_in_candidate = candidate_h - kh * dilation_h;
    int ih = h_in_candidate / stride_h;
    if (ih < 0 || ih >= in_h) continue;

    #pragma unroll
    for (int kw = offset_kw; kw < kw_end; kw += step_kw) {
      int w_in_candidate = candidate_w - kw * dilation_w;
      int iw = w_in_candidate / stride_w;
      if (iw < 0 || iw >= in_w) continue;

      int offset_pixel = ih * in_w + iw;
      // Compute weight offset related constants
      int weight_channel_offset = (oc - g * out_channels_per_group) * (kernel_h * kernel_w) + kh * kernel_w + kw;
      int weight_stride = out_channels_per_group * kernel_h * kernel_w;

      // Manual unroll over the input channel loop for this group
      int group_start = g * in_channels_per_group;
      int group_end = group_start + in_channels_per_group;
      int c = group_start;
      int group_count = in_channels_per_group;
      int remainder = group_count % 4;
      int limit = group_end - remainder;

      for (; c < limit; c += 4) {
        int idx0 = base_n + c * stride_c + offset_pixel;
        int idx1 = base_n + (c + 1) * stride_c + offset_pixel;
        int idx2 = base_n + (c + 2) * stride_c + offset_pixel;
        int idx3 = base_n + (c + 3) * stride_c + offset_pixel;

        int w_idx0 = c * weight_stride + weight_channel_offset;
        int w_idx1 = (c + 1) * weight_stride + weight_channel_offset;
        int w_idx2 = (c + 2) * weight_stride + weight_channel_offset;
        int w_idx3 = (c + 3) * weight_stride + weight_channel_offset;

        out_val += x[idx0] * weight[w_idx0]
                 + x[idx1] * weight[w_idx1]
                 + x[idx2] * weight[w_idx2]
                 + x[idx3] * weight[w_idx3];
      }
      for (; c < group_end; c++) {
        int idx = base_n + c * stride_c + offset_pixel;
        int w_idx = c * weight_stride + weight_channel_offset;
        out_val += x[idx] * weight[w_idx];
      }
    }
  }

  int out_index = n * (out_channels * out_h * out_w) +
                  oc * (out_h * out_w) +
                  oh * out_w + ow;
  output[out_index] = out_val;
}

// Host function wrapper for the CUDA kernel
at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int groups) {
  x = x.contiguous();
  weight = weight.contiguous();
  if (bias.has_value() && bias.value().defined())
    bias = bias.value().contiguous();

  const int batch = x.size(0);
  const int in_channels = x.size(1);
  const int in_h = x.size(2);
  const int in_w = x.size(3);

  const int kernel_h = weight.size(2);
  const int kernel_w = weight.size(3);
  const int out_channels_per_group = weight.size(1);
  const int out_channels = out_channels_per_group * groups;

  const int stride_h = stride[0];
  const int stride_w = stride[1];
  const int pad_h = padding[0];
  const int pad_w = padding[1];
  const int dilation_h = dilation[0];
  const int dilation_w = dilation[1];

  const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
  const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

  if (!bias.has_value() || !bias.value().defined()) {
    bias = at::zeros({out_channels}, weight.options());
  }

  auto output = at::zeros({batch, out_channels, out_h, out_w}, x.options());

  int in_channels_per_group = in_channels / groups;

  int total_threads = batch * out_channels * out_h * out_w;
  const int threads = 256;
  const int blocks = (total_threads + threads - 1) / threads;

  conv_transpose2d_kernel_manual_unroll<<<blocks, threads>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.value().data_ptr<float>(),
      output.data_ptr<float>(),
      batch, in_channels, in_h, in_w,
      out_channels, out_h, out_w,
      kernel_h, kernel_w,
      stride_h, stride_w,
      pad_h, pad_w,
      dilation_h, dilation_w,
      groups, in_channels_per_group, out_channels_per_group);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "2D Transposed Convolution with Manual Loop Unrolling (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"));
}
