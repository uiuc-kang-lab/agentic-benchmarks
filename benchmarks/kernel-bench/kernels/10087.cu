#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <algorithm>
#include <cmath>

namespace py = pybind11;

#define TILE_SIZE 16

// Helper: check if coordinate is within bounds
inline __device__ bool is_within_bounds(int coord, int lower_bound, int upper_bound) {
  return coord >= lower_bound && coord < upper_bound;
}

// Depthwise convolution kernel (operates on a sub-batch).
// Input: [batch, channels, in_h, in_w]
// Weight: [channels, 1, k, k]
// Output: [batch, channels, out_h, out_w]
template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch,          // batch size for this sub-batch
    int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k,
    int stride,
    int padding,
    int dilation) {

  // grid.z encodes (sub-batch index * channels)
  int linear_idx = blockIdx.z;
  int n = linear_idx / channels;
  int c = linear_idx % channels;

  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;

  if (oh < out_h && ow < out_w) {
    scalar_t sum = (bias != nullptr) ? bias[c] : static_cast<scalar_t>(0);
    for (int i = 0; i < k; ++i) {
      int ih = oh * stride - padding + i * dilation;
      bool ih_valid = is_within_bounds(ih, 0, in_h);
      for (int j = 0; j < k; ++j) {
        int iw = ow * stride - padding + j * dilation;
        bool iw_valid = is_within_bounds(iw, 0, in_w);
        int valid = ih_valid && iw_valid;
        int input_idx = n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
        int weight_idx = c * k * k + i * k + j;
        scalar_t in_val = valid ? input[input_idx] : static_cast<scalar_t>(0);
        sum += in_val * weight[weight_idx];
      }
    }
    int output_idx = n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
    output[output_idx] = sum;
  }
}

// Pointwise convolution kernel (1x1 convolution) for a sub-batch.
// Input: [batch, in_channels, h, w] -- output from depthwise stage
// Weight: [out_channels, in_channels]
// Output: [batch, out_channels, h, w]
template <typename scalar_t>
__global__ void pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch,           // sub-batch size
    int in_channels,
    int out_channels,
    int h, int w) {

  int linear_idx = blockIdx.z;
  int n = linear_idx / out_channels;
  int oc = linear_idx % out_channels;

  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;

  if (oh < h && ow < w) {
    scalar_t sum = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);
    for (int ic = 0; ic < in_channels; ++ic) {
      int input_idx = n * in_channels * h * w + ic * h * w + oh * w + ow;
      int weight_idx = oc * in_channels + ic;
      sum += input[input_idx] * weight[weight_idx];
    }
    int output_idx = n * out_channels * h * w + oc * h * w + oh * w + ow;
    output[output_idx] = sum;
  }
}

// Forward function using CUDA streams to overlap kernel execution with memory operations
// by splitting the input batch into sub-batches (pipeline).

torch::Tensor pipelined_forward_cuda(
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
  int in_channels = x.size(1);
  int in_h = x.size(2);
  int in_w = x.size(3);

  int k = depthwise_weight.size(2); // kernel size
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_channels = pointwise_weight.size(0);

  auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

  // Split the batch into sub-batches to pipeline kernel execution and (if any) memory transfers
  int num_streams = std::min(batch, 4);
  int chunk = (batch + num_streams - 1) / num_streams;

  std::vector<cudaStream_t> streams(num_streams);
  for (int s = 0; s < num_streams; s++) {
    cudaStreamCreate(&streams[s]);
  }

  dim3 block(TILE_SIZE, TILE_SIZE);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pipelined_forward_cuda", ([&] {
    auto* x_ptr = x.data_ptr<scalar_t>();
    auto* depthwise_weight_ptr = depthwise_weight.data_ptr<scalar_t>();
    auto* pointwise_weight_ptr = pointwise_weight.data_ptr<scalar_t>();
    auto* depthwise_bias_ptr = (depthwise_bias.defined() && depthwise_bias.numel() > 0) ? depthwise_bias.data_ptr<scalar_t>() : nullptr;
    auto* pointwise_bias_ptr = (pointwise_bias.defined() && pointwise_bias.numel() > 0) ? pointwise_bias.data_ptr<scalar_t>() : nullptr;
    auto* depthwise_output_ptr = depthwise_output.data_ptr<scalar_t>();
    auto* output_ptr = output.data_ptr<scalar_t>();

    for (int s = 0; s < num_streams; s++) {
      int batch_start = s * chunk;
      int current_chunk = std::min(chunk, batch - batch_start);
      if (current_chunk <= 0) break;

      // Compute pointers for the current sub-batch
      auto* x_sub = x_ptr + batch_start * in_channels * in_h * in_w;
      auto* depthwise_output_sub = depthwise_output_ptr + batch_start * in_channels * out_h * out_w;
      auto* output_sub = output_ptr + batch_start * out_channels * out_h * out_w;

      // Grid configuration for the depthwise kernel
      dim3 grid_depth((out_w + TILE_SIZE - 1) / TILE_SIZE,
                        (out_h + TILE_SIZE - 1) / TILE_SIZE,
                        current_chunk * in_channels);
      depthwise_conv2d_kernel<scalar_t><<<grid_depth, block, 0, streams[s]>>>(
          x_sub,
          depthwise_weight_ptr,
          depthwise_bias_ptr,
          depthwise_output_sub,
          current_chunk,
          in_channels,
          in_h, in_w,
          out_h, out_w,
          k,
          stride,
          padding,
          dilation);

      // Grid configuration for the pointwise kernel
      dim3 grid_point((out_w + TILE_SIZE - 1) / TILE_SIZE,
                      (out_h + TILE_SIZE - 1) / TILE_SIZE,
                      current_chunk * out_channels);
      pointwise_conv2d_kernel<scalar_t><<<grid_point, block, 0, streams[s]>>>(
          depthwise_output_sub,
          pointwise_weight_ptr,
          pointwise_bias_ptr,
          output_sub,
          current_chunk,
          in_channels,
          out_channels,
          out_h, out_w);

      // Optionally, asynchronous memory operations (e.g., cudaMemcpyAsync) can be added here to overlap
      // data transfers with computation if needed.
    }
  }));

  // Synchronize and destroy streams
  for (int s = 0; s < num_streams; s++) {
    cudaStreamSynchronize(streams[s]);
    cudaStreamDestroy(streams[s]);
  }

  return output;
}

// Helper to convert a py::object to a Torch tensor
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
at::Tensor pipelined_forward_wrapper(py::object x_obj,
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

  return pipelined_forward_cuda(x, depthwise_weight, pointwise_weight,
                                depthwise_bias, pointwise_bias,
                                stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pipelined_forward_wrapper, "Pipelined CUDA depthwise separable convolution forward with overlapped computation and memory transfers using CUDA streams");
}
