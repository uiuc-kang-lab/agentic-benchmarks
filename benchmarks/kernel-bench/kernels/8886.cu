#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

#define TILE_OC 32
#define TILE_SP 8

__device__ int gcd(int a, int b) {
  while(b != 0) {
    int t = b;
    b = a % b;
    a = t;
  }
  return a;
}

__device__ int my_min(int a, int b) {
  return a < b ? a : b;
}

// Optimized CUDA kernel to reduce warp divergence
__global__ void conv_transpose2d_kernel_no_divergence(
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

  int oc = blockIdx.x * TILE_OC + threadIdx.x;
  int sp_index = blockIdx.y * TILE_SP + threadIdx.y;
  int total_sp = out_h * out_w;
  int n = blockIdx.z;

  if (oc >= out_channels || sp_index >= total_sp)
      return;

  int oh = sp_index / out_w;
  int ow = sp_index % out_w;

  __shared__ float s_bias[TILE_OC];
  if (threadIdx.y == 0) {
    int oc_global = blockIdx.x * TILE_OC + threadIdx.x;
    if(oc_global < out_channels) {
      s_bias[threadIdx.x] = bias[oc_global];
    }
  }
  __syncthreads();

  float out_val = s_bias[threadIdx.x];

  int candidate_h = oh + pad_h;
  int candidate_w = ow + pad_w;

  int g = oc / out_channels_per_group;

  // Precompute limits
  int ih_start = (candidate_h + stride_h - 1) / stride_h;  
  int iw_start = (candidate_w + stride_w - 1) / stride_w;
  int ih_end = my_min((candidate_h - (kernel_h - 1) * dilation_h + stride_h - 1) / stride_h, in_h);
  int iw_end = my_min((candidate_w - (kernel_w - 1) * dilation_w + stride_w - 1) / stride_w, in_w);

  if (ih_start < 0 || iw_start < 0 || ih_start >= in_h || iw_start >= in_w) return;

  #pragma unroll
  for (int ih = ih_start; ih < ih_end; ++ih) {
    #pragma unroll
    for (int iw = iw_start; iw < iw_end; ++iw) {
      int h_offset = candidate_h - ih * stride_h;
      int w_offset = candidate_w - iw * stride_w;
      if (h_offset % dilation_h == 0 && w_offset % dilation_w == 0) {
        int kh = h_offset / dilation_h;
        int kw = w_offset / dilation_w;
        if (kh < 0 || kh >= kernel_h || kw < 0 || kw >= kernel_w) continue;

        #pragma unroll
        for (int c = g * in_channels_per_group; c < (g + 1) * in_channels_per_group; c++) {
          int x_index = n * (in_channels * in_h * in_w) +
                        c * (in_h * in_w) +
                        ih * in_w + iw;

          int weight_index = c * (out_channels_per_group * kernel_h * kernel_w) +
                             (oc - g * out_channels_per_group) * (kernel_h * kernel_w) +
                             kh * kernel_w + kw;

          out_val += x[x_index] * weight[weight_index];
        }  
      }
    }
  }

  int out_index = n * (out_channels * out_h * out_w) +
                  oc * (out_h * out_w) +
                  oh * out_w + ow;
  output[out_index] = out_val;
}

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

  dim3 block(TILE_OC, TILE_SP, 1);
  int total_spatial = out_h * out_w;
  dim3 grid((out_channels + TILE_OC - 1) / TILE_OC,
            (total_spatial + TILE_SP - 1) / TILE_SP,
            batch);

  conv_transpose2d_kernel_no_divergence<<<grid, block>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.value().data_ptr<float>(),
      output.data_ptr<float>(),
      batch,
      in_channels,
      in_h,
      in_w,
      out_channels,
      out_h,
      out_w,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      groups,
      in_channels_per_group,
      out_channels_per_group
  );

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "2D Transposed Convolution with Reduced Warp Divergence (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"));
}