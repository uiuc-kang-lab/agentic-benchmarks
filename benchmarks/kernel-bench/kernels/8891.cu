#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

// Inline device function to compute greatest common divisor
__device__ __forceinline__ int gcd_device(int a, int b) {
  while(b != 0) {
    int t = b;
    b = a % b;
    a = t;
  }
  return a;
}

// Inline device function for minimum
__device__ __forceinline__ int min_device(int a, int b) {
  return (a < b) ? a : b;
}

// Kernel using warp-level reduction for convolution accumulation
// Each warp computes one output element (n, oc, oh, ow)
// Block configuration: blockDim.x = 32 (one warp), blockDim.y = number of warps per block

__global__ void conv_transpose2d_kernel_reduction(
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
  // blockDim.x == 32, use threadIdx.y as warp id within block
  int lane = threadIdx.x;  // lane id within warp [0,31]
  int warp_in_block = threadIdx.y; // warp index in block
  
  // Compute global warp index across blocks in x dimension
  int warps_per_block = blockDim.y;  
  int global_warp = blockIdx.x * warps_per_block + warp_in_block;

  // Total number of output elements per sample
  int total_outputs = out_channels * out_h * out_w;
  if (global_warp >= total_outputs) return;

  // Map global_warp to output coordinates (for a given batch)
  int oc = global_warp / (out_h * out_w);
  int spatial_idx = global_warp % (out_h * out_w);
  int oh = spatial_idx / out_w;
  int ow = spatial_idx % out_w;

  // Batch index is obtained from blockIdx.z
  int n = blockIdx.z;

  // Load bias (one value per output channel)
  float bias_val = bias[oc];

  // Compute candidate positions (output position plus padding)
  int candidate_h = oh + pad_h;
  int candidate_w = ow + pad_w;

  // Compute valid kernel offsets and steps for height dimension
  int mod_h = candidate_h % stride_h;
  int offset_kh = -1;
  for (int k = 0; k < stride_h; k++) {
    if ((k * dilation_h) % stride_h == mod_h) {
      offset_kh = k;
      break;
    }
  }
  int step_kh = stride_h / gcd_device(stride_h, dilation_h);
  int kh_bound = candidate_h / dilation_h + 1;
  int kh_end = min_device(kernel_h, kh_bound);

  // Count valid kernel positions in height
  int valid_kh_count = 0;
  for (int kh = offset_kh; kh < kh_end; kh += step_kh) {
    valid_kh_count++;
  }

  // Compute valid kernel offsets and steps for width dimension
  int mod_w = candidate_w % stride_w;
  int offset_kw = -1;
  for (int k = 0; k < stride_w; k++) {
    if ((k * dilation_w) % stride_w == mod_w) {
      offset_kw = k;
      break;
    }
  }
  int step_kw = stride_w / gcd_device(stride_w, dilation_w);
  int kw_bound = candidate_w / dilation_w + 1;
  int kw_end = min_device(kernel_w, kw_bound);

  // Count valid kernel positions in width
  int valid_kw_count = 0;
  for (int kw = offset_kw; kw < kw_end; kw += step_kw) {
    valid_kw_count++;
  }

  // Total number of contributions for this output element:
  // Over input channels (within group) and valid kernel positions
  int total_iters = in_channels_per_group * valid_kh_count * valid_kw_count;
  float sum = 0.0f;

  // Determine group for this output channel
  int g = oc / out_channels_per_group;
  int group_offset = g * in_channels_per_group;  // starting index for input channels in this group

  // Each thread in the warp processes a subset of iterations
  for (int iter = lane; iter < total_iters; iter += 32) {
    int c_idx = iter / (valid_kh_count * valid_kw_count);  // relative channel index [0, in_channels_per_group)
    int rem = iter % (valid_kh_count * valid_kw_count);
    int kh_idx = rem / valid_kw_count;  // index for valid kernel height
    int kw_idx = rem % valid_kw_count;  // index for valid kernel width

    int kh = offset_kh + kh_idx * step_kh;
    int kw = offset_kw + kw_idx * step_kw;

    // Compute the corresponding input indices
    int h_in_candidate = candidate_h - kh * dilation_h;
    int w_in_candidate = candidate_w - kw * dilation_w;
    int ih = h_in_candidate / stride_h;
    int iw = w_in_candidate / stride_w;

    // Boundary check
    if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) continue;

    int c = group_offset + c_idx;  // global input channel index
    int x_index = n * (in_channels * in_h * in_w) + c * (in_h * in_w) + ih * in_w + iw;
    int weight_index = c * (out_channels_per_group * kernel_h * kernel_w) +
                       (oc - g * out_channels_per_group) * (kernel_h * kernel_w) +
                       kh * kernel_w + kw;

    sum += x[x_index] * weight[weight_index];
  }

  // Warp-level reduction using __shfl_down_sync
  unsigned int mask = 0xffffffff;
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(mask, sum, offset);
  }

  // The first lane writes the result
  if (lane == 0) {
    float out_val = bias_val + sum;  // add bias
    int out_index = n * (out_channels * out_h * out_w) +
                    oc * (out_h * out_w) +
                    oh * out_w + ow;
    output[out_index] = out_val;
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

  // Extract dimensions
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
  const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
  const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

  // If bias is not provided, create a zero tensor
  if (!bias.has_value() || !bias.value().defined()) {
    bias = at::zeros({out_channels}, weight.options());
  }

  auto output = at::zeros({batch, out_channels, out_h, out_w}, x.options());

  int in_channels_per_group = in_channels / groups;

  // Total number of output elements per sample
  int total_out = out_channels * out_h * out_w;
  
  // Configure kernel: each warp (32 threads) computes one output element.
  // Choose number of warps per block, e.g., 8 warps per block.
  int warps_per_block = 8;
  dim3 block(32, warps_per_block, 1);
  int grid_x = (total_out + warps_per_block - 1) / warps_per_block;
  dim3 grid(grid_x, 1, batch);

  conv_transpose2d_kernel_reduction<<<grid, block>>>(
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
  m.def("forward", &forward, "2D Transposed Convolution with Warp-level Reduction (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"));
}
