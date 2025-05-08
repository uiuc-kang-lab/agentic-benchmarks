/*
This CUDA extension implements a fused conv_transpose3d operation. It combines
both the native, highly optimized ATen (cuDNN-based) conv_transpose3d and a
custom CUDA kernel for transposed convolution. The fused kernel supports both
grouped and non-grouped cases via a unified device function and selects the
optimal path at runtime based on the total output volume.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Unified device function for computing a single output value for conv_transpose3d.
// This function handles both grouped and non-grouped convolutions.

__device__ __forceinline__
float fused_compute_conv_transpose3d(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int b, int oc,
    int out_d, int out_h, int out_w,
    int in_channels,
    int iD, int iH, int iW,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int groups,
    int out_channels) {
  float sum = 0.0f;
  if (groups == 1) {
    // Non-grouped convolution
    for (int ic = 0; ic < in_channels; ++ic) {
      for (int kd = 0; kd < kD; ++kd) {
        int id = out_d + pad_d - kd;
        if (id % stride_d != 0) continue;
        id /= stride_d;
        if (id < 0 || id >= iD) continue;
        for (int kh = 0; kh < kH; ++kh) {
          int ih = out_h + pad_h - kh;
          if (ih % stride_h != 0) continue;
          ih /= stride_h;
          if (ih < 0 || ih >= iH) continue;
          for (int kw = 0; kw < kW; ++kw) {
            int iw = out_w + pad_w - kw;
            if (iw % stride_w != 0) continue;
            iw /= stride_w;
            if (iw < 0 || iw >= iW) continue;
            int input_idx = (((b * in_channels + ic) * iD + id) * iH + ih) * iW + iw;
            int weight_idx = ((((ic) * out_channels + oc) * kD + kd) * kH + kh) * kW + kw;
            sum += __ldg(&input[input_idx]) * weight[weight_idx];
          }
        }
      }
    }
  } else {
    // Grouped convolution
    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;
    int group = oc / out_channels_per_group;
    for (int ic = group * in_channels_per_group;
         ic < (group + 1) * in_channels_per_group; ++ic) {
      for (int kd = 0; kd < kD; ++kd) {
        int id = out_d + pad_d - kd;
        if (id % stride_d != 0) continue;
        id /= stride_d;
        if (id < 0 || id >= iD) continue;
        for (int kh = 0; kh < kH; ++kh) {
          int ih = out_h + pad_h - kh;
          if (ih % stride_h != 0) continue;
          ih /= stride_h;
          if (ih < 0 || ih >= iH) continue;
          for (int kw = 0; kw < kW; ++kw) {
            int iw = out_w + pad_w - kw;
            if (iw % stride_w != 0) continue;
            iw /= stride_w;
            if (iw < 0 || iw >= iW) continue;
            int input_idx = (((b * in_channels + ic) * iD + id) * iH + ih) * iW + iw;
            int weight_ic = ic - group * in_channels_per_group;
            int oc_local = oc % out_channels_per_group;
            int weight_idx = ((((weight_ic) * out_channels_per_group + oc_local) * kD + kd) * kH + kh) * kW + kw;
            sum += __ldg(&input[input_idx]) * weight[weight_idx];
          }
        }
      }
    }
  }
  return sum;
}

// Optimized CUDA kernel launching one thread per output element.
// It uses the fused_compute_conv_transpose3d to accumulate the convolution sum and integrates
// bias addition. The kernel supports both grouped and non-grouped convolutions.

__global__ void fused_conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // may be nullptr
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int iD, int iH, int iW,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int outD, int outH, int outW,
    int groups) {

  int total = batch * out_channels * outD * outH * outW;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  // Decode linear index into (b, oc, d, h, w)
  int w = idx % outW;
  int tmp = idx / outW;
  int h = tmp % outH;
  tmp /= outH;
  int d = tmp % outD;
  tmp /= outD;
  int oc = tmp % out_channels;
  int b = tmp / out_channels;

  float sum = fused_compute_conv_transpose3d(
      input, weight,
      b, oc,
      d, h, w,
      in_channels, iD, iH, iW,
      kD, kH, kW,
      stride_d, stride_h, stride_w,
      pad_d, pad_h, pad_w,
      groups, out_channels);

  // Fuse bias addition using a fast inline check
  sum += (bias != nullptr) ? bias[oc] : 0.0f;

  int out_idx = (((b * out_channels + oc) * outD + d) * outH + h) * outW + w;
  output[out_idx] = sum;
}

// Hybrid forward function.
// For large problem sizes, it falls back to the ATen conv_transpose3d (which can leverage cuDNN).
// For smaller sizes, it launches our custom CUDA kernel.

torch::Tensor fused_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

  // Input dimensions: [batch, in_channels, iD, iH, iW]
  int batch = x.size(0);
  int in_channels = x.size(1);
  int iD = x.size(2);
  int iH = x.size(3);
  int iW = x.size(4);

  // Weight dimensions: For groups==1: [in_channels, out_channels, kD, kH, kW]
  // For grouped convolution, shape is [in_channels_per_group, out_channels_per_group, kD, kH, kW]
  int kD = weight.size(2);
  int kH = weight.size(3);
  int kW = weight.size(4);

  int stride_d = stride[0];
  int stride_h = stride[1];
  int stride_w = stride[2];
  int pad_d = padding[0];
  int pad_h = padding[1];
  int pad_w = padding[2];
  int opad_d = output_padding[0];
  int opad_h = output_padding[1];
  int opad_w = output_padding[2];

  // Compute output dimensions for transposed convolution
  int outD = (iD - 1) * stride_d - 2 * pad_d + kD + opad_d;
  int outH = (iH - 1) * stride_h - 2 * pad_h + kH + opad_h;
  int outW = (iW - 1) * stride_w - 2 * pad_w + kW + opad_w;

  int out_channels = (groups == 1) ? weight.size(1) : weight.size(1) * groups;

  // Determine whether to use the custom kernel or fallback to ATen's version
  int total_elements = batch * out_channels * outD * outH * outW;
  const int threshold = 1024 * 1024; // Arbitrary threshold
  if (total_elements > threshold) {
    // Fallback: Call ATen's conv_transpose3d (which may leverage cuDNN) for big workloads
    std::vector<int64_t> dilation = {1, 1, 1};
    return at::conv_transpose3d(x,
                                weight,
                                bias ? *bias : torch::Tensor(),
                                stride,
                                padding,
                                output_padding,
                                groups,
                                dilation);
  }

  auto options = x.options();
  auto output = torch::zeros({batch, out_channels, outD, outH, outW}, options);

  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  fused_conv_transpose3d_kernel<<<blocks, threads>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
      output.data_ptr<float>(),
      batch, in_channels, out_channels,
      iD, iH, iW,
      kD, kH, kW,
      stride_d, stride_h, stride_w,
      pad_d, pad_h, pad_w,
      outD, outH, outW,
      groups);

  cudaDeviceSynchronize();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_forward, "Fused and Hybrid ConvTranspose3d forward function",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = nullptr,
        py::arg("stride"),
        py::arg("padding"),
        py::arg("output_padding"),
        py::arg("groups"));
}
