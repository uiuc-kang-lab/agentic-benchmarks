#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized CUDA kernel with improved thread and block mapping
__global__ void optimized_conv_transpose3d_kernel(
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
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) return;

  int w = index % outW; 
  index /= outW;
  int h = index % outH; 
  index /= outH;
  int d = index % outD; 
  index /= outD;
  int oc = index % out_channels; 
  index /= out_channels;
  int b = index;

  float sum = 0.0f;
  if (groups == 1) {
    for (int ic = 0; ic < in_channels; ++ic) {
      for (int kd = 0; kd < kD; ++kd) {
        int id = d + pad_d - kd;
        if (id % stride_d != 0) continue;
        id /= stride_d;
        if (id < 0 || id >= iD) continue;
        for (int kh = 0; kh < kH; ++kh) {
          int ih = h + pad_h - kh;
          if (ih % stride_h != 0) continue;
          ih /= stride_h;
          if (ih < 0 || ih >= iH) continue;
          for (int kw = 0; kw < kW; ++kw) {
            int iw = w + pad_w - kw;
            if (iw % stride_w != 0) continue;
            iw /= stride_w;
            if (iw < 0 || iw >= iW) continue;
            int input_idx = (((b * in_channels + ic) * iD + id) * iH + ih) * iW + iw;
            int weight_idx = ((((ic) * out_channels + oc) * kD + kd) * kH + kh) * kW + kw;
            sum += input[input_idx] * weight[weight_idx];
          }
        }
      }
    }
  } else {
    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;
    int group = oc / out_channels_per_group;
    for (int ic = group * in_channels_per_group; ic < (group + 1) * in_channels_per_group; ++ic) {
      for (int kd = 0; kd < kD; ++kd) {
        int id = d + pad_d - kd;
        if (id % stride_d != 0) continue;
        id /= stride_d;
        if (id < 0 || id >= iD) continue;
        for (int kh = 0; kh < kH; ++kh) {
          int ih = h + pad_h - kh;
          if (ih % stride_h != 0) continue;
          ih /= stride_h;
          if (ih < 0 || ih >= iH) continue;
          for (int kw = 0; kw < kW; ++kw) {
            int iw = w + pad_w - kw;
            if (iw % stride_w != 0) continue;
            iw /= stride_w;
            if (iw < 0 || iw >= iW) continue;
            int input_idx = (((b * in_channels + ic) * iD + id) * iH + ih) * iW + iw;
            int weight_ic = ic - group * in_channels_per_group;
            int oc_local = oc % out_channels_per_group;
            int weight_idx = ((((weight_ic) * out_channels_per_group + oc_local) * kD + kd) * kH + kh) * kW + kw;
            sum += input[input_idx] * weight[weight_idx];
          }
        }
      }
    }
  }

  if (bias != nullptr) {
    sum += bias[oc];
  }

  int out_idx = (((b * out_channels + oc) * outD + d) * outH + h) * outW + w;
  output[out_idx] = sum;
}

// Host forward function
torch::Tensor optimized_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

  int batch = x.size(0);
  int in_channels = x.size(1);
  int iD = x.size(2);
  int iH = x.size(3);
  int iW = x.size(4);

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

  int outD = (iD - 1) * stride_d - 2 * pad_d + kD + opad_d;
  int outH = (iH - 1) * stride_h - 2 * pad_h + kH + opad_h;
  int outW = (iW - 1) * stride_w - 2 * pad_w + kW + opad_w;

  int out_channels = (groups == 1) ? weight.size(1) : weight.size(1) * groups;

  auto options = x.options();
  auto output = torch::zeros({batch, out_channels, outD, outH, outW}, options);

  dim3 threads(8, 8, 8);
  dim3 blocks((outW + threads.x - 1) / threads.x,
               (outH + threads.y - 1) / threads.y,
               (outD + threads.z - 1) / threads.z);

  optimized_conv_transpose3d_kernel<<<blocks, threads>>>(
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
  m.def("forward", &optimized_forward, "Optimized ConvTranspose3d forward function",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = nullptr,
        py::arg("stride"),
        py::arg("padding"),
        py::arg("output_padding"),
        py::arg("groups"));
}
