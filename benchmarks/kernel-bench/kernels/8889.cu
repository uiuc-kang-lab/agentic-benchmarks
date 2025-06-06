#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

// Utility device functions

__device__ __forceinline__ int gcd_device(int a, int b) {
  while (b != 0) {
    int t = b;
    b = a % b;
    a = t;
  }
  return a;
}

__device__ __forceinline__ int min_device(int a, int b) {
  return (a < b) ? a : b;
}

// Kernel implementing warp-level reduction for conv_transposed2d
// Each warp computes one output element by partitioning the reduction over the (kh, kw, c) loops.

__global__ void conv_transpose2d_kernel_warp(
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

  // Each warp (32 threads) computes one output element
  // Compute global warp id from block and thread indices
  int warp_id_in_block = threadIdx.y;  // which warp within the block
  int lane = threadIdx.x;              // lane id in the warp (0-31)
  int global_warp = blockIdx.x * blockDim.y + warp_id_in_block;
  
  // Total number of output spatial elements per batch
  int total_out = out_channels * out_h * out_w;
  if (global_warp >= total_out) return;

  // Decode output element indices from global warp id
  int oc = global_warp / (out_h * out_w);
  int rem = global_warp % (out_h * out_w);
  int oh = rem / out_w;
  int ow = rem % out_w;

  // Batch index from grid.z
  int n = blockIdx.z;

  // Candidate positions (adding padding)
  int candidate_h = oh + pad_h;
  int candidate_w = ow + pad_w;

  // Compute valid kernel offset for height dimension
  int offset_kh = -1;
  int mod_h = candidate_h % stride_h;
  for (int i = 0; i < stride_h; i++) {
    if ((i * dilation_h) % stride_h == mod_h) {
      offset_kh = i;
      break;
    }
  }
  int step_kh = stride_h / gcd_device(stride_h, dilation_h);
  int kh_bound = candidate_h / dilation_h + 1;
  int kh_end = min_device(kernel_h, kh_bound);

  // Compute valid kernel offset for width dimension
  int offset_kw = -1;
  int mod_w = candidate_w % stride_w;
  for (int i = 0; i < stride_w; i++) {
    if ((i * dilation_w) % stride_w == mod_w) {
      offset_kw = i;
      break;
    }
  }
  int step_kw = stride_w / gcd_device(stride_w, dilation_w);
  int kw_bound = candidate_w / dilation_w + 1;
  int kw_end = min_device(kernel_w, kw_bound);

  // Determine group index based on output channel
  int g = oc / out_channels_per_group;
  int group_in_start = g * in_channels_per_group;
  int group_in_end = (g + 1) * in_channels_per_group;

  // Each warp partitions the reduction over its 32 lanes.
  float partial_sum = 0.0f;
  int iter = 0;  // iteration counter across the (kh, kw, c) loops

  // Loop over kernel height offsets
  for (int kh = offset_kh; kh < kh_end; kh += step_kh) {
    int h_in_candidate = candidate_h - kh * dilation_h;
    // Check validity: candidate must be divisible by stride and within input height
    bool valid_h = ((h_in_candidate % stride_h) == 0);
    int ih = valid_h ? (h_in_candidate / stride_h) : -1;
    valid_h = valid_h && (ih >= 0 && ih < in_h);

    // Loop over kernel width offsets
    for (int kw = offset_kw; kw < kw_end; kw += step_kw) {
      int w_in_candidate = candidate_w - kw * dilation_w;
      bool valid_w = ((w_in_candidate % stride_w) == 0);
      int iw = valid_w ? (w_in_candidate / stride_w) : -1;
      valid_w = valid_w && (iw >= 0 && iw < in_w);

      bool valid = valid_h && valid_w;

      // Loop over input channels for the group
      for (int c = group_in_start; c < group_in_end; c++) {
        // Distribute iterations among warp lanes
        if ((iter % 32) == lane) {
          if (valid) {
            int x_index = n * (in_channels * in_h * in_w) +
                          c * (in_h * in_w) +
                          ih * in_w + iw;
            int weight_index = c * (out_channels_per_group * kernel_h * kernel_w) +
                               (oc - g * out_channels_per_group) * (kernel_h * kernel_w) +
                               kh * kernel_w + kw;
            partial_sum += x[x_index] * weight[weight_index];
          }
        }
        iter++;
      }
    }
  }

  // Warp-level reduction using __shfl_down_sync
  unsigned mask = 0xffffffff;
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum += __shfl_down_sync(mask, partial_sum, offset);
  }

  // Lane 0 writes the result, adding the bias
  if (lane == 0) {
    partial_sum += bias[oc];
    int out_index = n * (out_channels * out_h * out_w) +
                    oc * (out_h * out_w) +
                    oh * out_w + ow;
    output[out_index] = partial_sum;
  }
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

  // Ensure inputs are contiguous
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

  // Compute output dimensions for conv_transpose2d
  const int out_h = (x.size(2) - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
  const int out_w = (x.size(3) - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

  if (!bias.has_value() || !bias.value().defined()) {
    bias = at::zeros({out_channels}, weight.options());
  }

  auto output = at::zeros({batch, out_channels, out_h, out_w}, x.options());
  int in_channels_per_group = in_channels / groups;

  // Each warp (32 threads) computes one output element
  // Total output elements per batch
  int total_out = out_channels * out_h * out_w;

  // Choose number of warps per block; here we use 8 warps per block as an example
  int warps_per_block = 8;
  dim3 block(32, warps_per_block, 1);
  // grid.x: number of warps needed to cover total output elements
  dim3 grid((total_out + warps_per_block - 1) / warps_per_block, 1, batch);

  conv_transpose2d_kernel_warp<<<grid, block>>>(
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
      out_channels_per_group);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "2D Transposed Convolution with Warp-level Reduction (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"));
}
