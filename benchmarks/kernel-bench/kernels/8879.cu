#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>
#include <stdio.h>

// Define tile sizes for output channels and spatial dimensions
#define TILE_OC 32
#define TILE_SP 8

// Inline device function to compute greatest common divisor
__device__ int gcd_device(int a, int b) {
  while(b != 0) {
    int t = b;
    b = a % b;
    a = t;
  }
  return a;
}

// Inline device function for minimum
__device__ int min_device(int a, int b) {
  return (a < b) ? a : b;
}

// 2D Tiled CUDA kernel for 2D transposed convolution
// Each block computes a tile of outputs for a given batch index.
// The tile is organized along output channels (x-dimension) and spatial (flattened oh*ow in y-dimension).
// Bias values for the tile are loaded into shared memory with a single __syncthreads() after loading.

__global__ void conv_transpose2d_kernel_tiled(
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

  // Determine output channel and spatial indices using 2D tiling
  int oc = blockIdx.x * TILE_OC + threadIdx.x;  // output channel
  int sp_index = blockIdx.y * TILE_SP + threadIdx.y;  // flattened spatial index for (oh, ow)
  int total_sp = out_h * out_w;

  int n = blockIdx.z; // batch index

  if (oc >= out_channels || sp_index >= total_sp)
      return;

  int oh = sp_index / out_w;
  int ow = sp_index % out_w;

  // Load bias for this tile (for output channels) into shared memory
  __shared__ float s_bias[TILE_OC];
  if (threadIdx.y == 0) {
    int oc_global = blockIdx.x * TILE_OC + threadIdx.x;
    if(oc_global < out_channels) {
      s_bias[threadIdx.x] = bias[oc_global];
    }
  }
  __syncthreads(); // Synchronize only once after loading shared bias

  float out_val = s_bias[threadIdx.x];

  // Compute candidate positions by adding padding
  int candidate_h = oh + pad_h;
  int candidate_w = ow + pad_w;

  // Compute valid kernel offsets for height dimension
  int offset_kh = -1;
  int mod_h = candidate_h % stride_h;
  for (int k = 0; k < stride_h; k++) {
    if ((k * dilation_h) % stride_h == mod_h) {
      offset_kh = k;
      break;
    }
  }
  int step_kh = stride_h / gcd_device(stride_h, dilation_h);
  int kh_bound = candidate_h / dilation_h + 1;
  int kh_end = min_device(kernel_h, kh_bound);

  // Compute valid kernel offsets for width dimension
  int offset_kw = -1;
  int mod_w = candidate_w % stride_w;
  for (int k = 0; k < stride_w; k++) {
    if ((k * dilation_w) % stride_w == mod_w) {
      offset_kw = k;
      break;
    }
  }
  int step_kw = stride_w / gcd_device(stride_w, dilation_w);
  int kw_bound = candidate_w / dilation_w + 1;
  int kw_end = min_device(kernel_w, kw_bound);

  // Determine group for this output channel
  int g = oc / out_channels_per_group;

  // Iterate over the valid kernel positions with loop unrolling
  #pragma unroll
  for (int kh = offset_kh; (kh >= 0) && (kh < kh_end); kh += step_kh) {
    int h_in_candidate = candidate_h - kh * dilation_h;
    int ih = h_in_candidate / stride_h;
    if (ih < 0 || ih >= in_h) continue;

    #pragma unroll
    for (int kw = offset_kw; (kw >= 0) && (kw < kw_end); kw += step_kw) {
      int w_in_candidate = candidate_w - kw * dilation_w;
      int iw = w_in_candidate / stride_w;
      if (iw < 0 || iw >= in_w) continue;

      // Loop over corresponding input channels within the same group
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

  int out_index = n * (out_channels * out_h * out_w) +
                  oc * (out_h * out_w) +
                  oh * out_w + ow;
  output[out_index] = out_val;
}

// Host function wrapping the CUDA kernel
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

  // Set up 2D block and 3D grid dimensions
  dim3 block(TILE_OC, TILE_SP, 1);
  int total_spatial = out_h * out_w;
  dim3 grid((out_channels + TILE_OC - 1) / TILE_OC,
            (total_spatial + TILE_SP - 1) / TILE_SP,
            batch);

  conv_transpose2d_kernel_tiled<<<grid, block>>>(
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
  m.def("forward", &forward, "2D Transposed Convolution with Tiling and Shared Bias (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"));
}
