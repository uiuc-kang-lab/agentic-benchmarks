#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define TILE_SIZE 16
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4

// Depthwise convolution kernel with branchless bounds checking (unchanged from previous branchless version)
template <typename scalar_t>
__global__ void branchless_depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,   // [batch, channels, in_h, in_w]
    const scalar_t* __restrict__ weight,  // [channels, 1, k, k]
    const scalar_t* __restrict__ bias,    // [channels] or nullptr
    scalar_t* __restrict__ output,        // [batch, channels, out_h, out_w]
    int batch,
    int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k,
    int stride,
    int padding,
    int dilation) {
  
  // Each thread is responsible for one output pixel in a given (batch, channel)
  int linear_idx = blockIdx.z;
  int n = linear_idx / channels;
  int c = linear_idx % channels;

  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;

  if (oh < out_h && ow < out_w) {
    scalar_t sum = (bias != nullptr) ? bias[c] : static_cast<scalar_t>(0);
    
    // Loop over the kernel window
    for (int i = 0; i < k; i++) {
      int ih = oh * stride - padding + i * dilation;
      int valid_ih = ((unsigned)ih < (unsigned)in_h);
      int safe_ih = valid_ih ? ih : 0;
      
      for (int j = 0; j < k; j++) {
        int iw = ow * stride - padding + j * dilation;
        int valid_iw = ((unsigned)iw < (unsigned)in_w);
        int safe_iw = valid_iw ? iw : 0;
        int valid = valid_ih * valid_iw;

        int input_idx = n * channels * in_h * in_w + c * in_h * in_w + safe_ih * in_w + safe_iw;
        int weight_idx = c * k * k + i * k + j;
        sum += valid * input[input_idx] * weight[weight_idx];
      }
    }
    
    int output_idx = n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
    output[output_idx] = sum;
  }
}

// Warp-level optimized pointwise convolution kernel using __shfl_down_sync for reduction
// Each warp computes one output pixel
template <typename scalar_t>
__global__ void warp_pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,   // [batch, in_channels, h, w] from depthwise output
    const scalar_t* __restrict__ weight,  // [out_channels, in_channels]
    const scalar_t* __restrict__ bias,    // [out_channels] or nullptr
    scalar_t* __restrict__ output,        // [batch, out_channels, h, w]
    int batch,
    int in_channels,
    int out_channels,
    int h,
    int w,
    int total_out) {  // total output elements: batch * out_channels * h * w

  // Each warp processes one output element
  // Calculate global warp id: each block has WARPS_PER_BLOCK warps, where blockDim.x = WARP_SIZE and blockDim.y = WARPS_PER_BLOCK
  int warp_id = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;
  int lane = threadIdx.x; // lane id in warp
  
  if (warp_id < total_out) {
    // Decode warp_id into (n, oc, oh, ow)
    int out_hw = h * w;
    int oc_hw = out_channels * out_hw;
    int n = warp_id / oc_hw;
    int rem = warp_id % oc_hw;
    int oc = rem / out_hw;
    int pos = rem % out_hw;
    int oh = pos / w;
    int ow = pos % w;

    scalar_t partial_sum = 0;
    // Loop over in_channels with stride equal to warp size
    for (int ic = lane; ic < in_channels; ic += WARP_SIZE) {
      int input_idx = n * (in_channels * h * w) + ic * (h * w) + oh * w + ow;
      int weight_idx = oc * in_channels + ic;
      partial_sum += input[input_idx] * weight[weight_idx];
    }

    // Warp-level reduction using __shfl_down_sync
    // Full mask for active lanes
    unsigned int mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }

    if (lane == 0) {
      if (bias != nullptr) {
        partial_sum += bias[oc];
      }
      int output_idx = n * (out_channels * h * w) + oc * (h * w) + oh * w + ow;
      output[output_idx] = partial_sum;
    }
  }
}

// Core CUDA forward function
torch::Tensor forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& depthwise_weight,
    const torch::Tensor& pointwise_weight,
    const torch::Tensor& depthwise_bias,
    const torch::Tensor& pointwise_bias,
    int stride,
    int padding,
    int dilation) {

  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
  TORCH_CHECK(depthwise_weight.is_cuda(), "Depthwise weight must be a CUDA tensor");
  TORCH_CHECK(pointwise_weight.is_cuda(), "Pointwise weight must be a CUDA tensor");
  if (depthwise_bias.defined() && depthwise_bias.numel() > 0)
    TORCH_CHECK(depthwise_bias.is_cuda(), "Depthwise bias must be a CUDA tensor if provided");
  if (pointwise_bias.defined() && pointwise_bias.numel() > 0)
    TORCH_CHECK(pointwise_bias.is_cuda(), "Pointwise bias must be a CUDA tensor if provided");

  int batch = x.size(0);
  int channels = x.size(1);
  int in_h = x.size(2);
  int in_w = x.size(3);
  
  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  // Allocate tensor for depthwise convolution output
  auto depthwise_output = torch::empty({batch, channels, out_h, out_w}, x.options());

  // Set up grid and block for branchless depthwise convolution
  dim3 block_dw(TILE_SIZE, TILE_SIZE);
  dim3 grid_dw((out_w + TILE_SIZE - 1) / TILE_SIZE,
               (out_h + TILE_SIZE - 1) / TILE_SIZE,
               batch * channels);

  const void* depthwise_bias_ptr = (depthwise_bias.defined() && depthwise_bias.numel() > 0) ? depthwise_bias.data_ptr() : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "branchless_depthwise_conv2d_cuda", ([&] {
    branchless_depthwise_conv2d_kernel<scalar_t><<<grid_dw, block_dw>>>(
        x.data_ptr<scalar_t>(),
        depthwise_weight.data_ptr<scalar_t>(),
        reinterpret_cast<const scalar_t*>(depthwise_bias_ptr),
        depthwise_output.data_ptr<scalar_t>(),
        batch,
        channels,
        in_h, in_w,
        out_h, out_w,
        k,
        stride,
        padding,
        dilation);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Depthwise kernel launch error: %s\n", cudaGetErrorString(err));
  }

  // Pointwise convolution: weight shape is [out_channels, in_channels]
  int out_channels = pointwise_weight.size(0);
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

  // Total number of output elements for pointwise conv
  int total_out = batch * out_channels * out_h * out_w;
  
  // Configure grid and block for warp-level pointwise convolution kernel
  // Each block has WARPS_PER_BLOCK warps, with each warp having WARP_SIZE threads
  dim3 block_wp(WARP_SIZE, WARPS_PER_BLOCK);
  int blocks_wp = (total_out + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  dim3 grid_wp(blocks_wp);

  const void* pointwise_bias_ptr = (pointwise_bias.defined() && pointwise_bias.numel() > 0) ? pointwise_bias.data_ptr() : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "warp_pointwise_conv2d_cuda", ([&] {
    warp_pointwise_conv2d_kernel<scalar_t><<<grid_wp, block_wp>>>(
        depthwise_output.data_ptr<scalar_t>(),
        pointwise_weight.data_ptr<scalar_t>(),
        reinterpret_cast<const scalar_t*>(pointwise_bias_ptr),
        output.data_ptr<scalar_t>(),
        batch,
        channels,  // in_channels for pointwise conv equals depthwise output channels
        out_channels,
        out_h, out_w,
        total_out);
  }));

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Pointwise kernel launch error: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Helper: convert a py::object to an at::Tensor
at::Tensor toTensor(const py::object& obj) {
  if (obj.is_none()) {
    return at::Tensor();
  }
  try {
    return obj.cast<at::Tensor>();
  } catch (const py::cast_error& e) {
    if (py::hasattr(obj, "data")) {
      return obj.attr("data").cast<at::Tensor>();
    }
    throw std::runtime_error("Expected a torch Tensor or Parameter.");
  }
}

// Wrapper function exposed to Python
at::Tensor forward_wrapper(py::object x_obj,
                           py::object depthwise_weight_obj,
                           py::object pointwise_weight_obj,
                           py::object depthwise_bias_obj,
                           py::object pointwise_bias_obj,
                           int stride,
                           int padding,
                           int dilation) {
  auto x = toTensor(x_obj);
  auto depthwise_weight = toTensor(depthwise_weight_obj);
  auto pointwise_weight = toTensor(pointwise_weight_obj);
  auto depthwise_bias = toTensor(depthwise_bias_obj);
  auto pointwise_bias = toTensor(pointwise_bias_obj);
  
  return forward_cuda(x, depthwise_weight, pointwise_weight,
                      depthwise_bias, pointwise_bias,
                      stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_wrapper, "CUDA depthwise separable convolution forward with warp-level reduction in pointwise conv");
}
