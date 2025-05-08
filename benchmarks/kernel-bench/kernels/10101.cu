#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Block tile size for spatial dimensions
#define TILE_SIZE 16

// Depthwise convolution kernel leveraging shared memory for both input tile and filter weights.
// Each block is responsible for computing a tile of the output for one (batch, channel).

template <typename scalar_t>
__global__ void shared_mem_depthwise_conv2d_kernel(
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

  // Determine which image and channel this block handles
  int linear_idx = blockIdx.z;  // linear index over (batch, channel)
  int n = linear_idx / channels;
  int c = linear_idx % channels;

  // Calculate the starting index in the output for this block tile
  int block_out_x = blockIdx.x * blockDim.x;  // output x index for this block
  int block_out_y = blockIdx.y * blockDim.y;  // output y index for this block

  // Dimensions of the shared memory tile for the input region.
  // For each output pixel, a receptive field of size (k x k) is applied with dilation.
  int tile_width = blockDim.x * stride + (k - stride);
  int tile_height = blockDim.y * stride + (k - stride);

  // Compute starting coordinates in the input feature map for this tile.
  int in_start_x = block_out_x * stride - padding;
  int in_start_y = block_out_y * stride - padding;

  // Allocate dynamic shared memory:
  // First part: shared memory for the input tile of size tile_width * tile_height
  // Second part: shared memory for the filter weights of this channel (size k * k)
  extern __shared__ char smem[];
  scalar_t* s_input = reinterpret_cast<scalar_t*>(smem);
  int input_tile_size = tile_width * tile_height;
  scalar_t* s_weight = s_input + input_tile_size;

  // Load the filter weights into shared memory. They are reused by all threads in the block.
  int t_idx = threadIdx.y * blockDim.x + threadIdx.x;
  int weight_elements = k * k;
  for (int i = t_idx; i < weight_elements; i += blockDim.x * blockDim.y) {
      s_weight[i] = weight[c * weight_elements + i];
  }

  // Load the input tile into shared memory.
  int total_tile = tile_width * tile_height;
  for (int i = t_idx; i < total_tile; i += blockDim.x * blockDim.y) {
      int local_x = i % tile_width;
      int local_y = i / tile_width;
      int global_x = in_start_x + local_x;
      int global_y = in_start_y + local_y;
      scalar_t val = 0;
      if (global_x >= 0 && global_x < in_w && global_y >= 0 && global_y < in_h) {
          int input_idx = n * channels * in_h * in_w + c * in_h * in_w + global_y * in_w + global_x;
          val = input[input_idx];
      }
      s_input[i] = val;
  }
  __syncthreads();

  // Determine the output pixel coordinates for this thread
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ow = block_out_x + tx;
  int oh = block_out_y + ty;

  if (ow < out_w && oh < out_h) {
      // In shared memory, the top-left of the receptive field for this output pixel
      // corresponds to (tx * stride, ty * stride)
      int local_x = tx * stride;
      int local_y = ty * stride;
      scalar_t sum = 0;
      for (int i = 0; i < k; i++) {
          for (int j = 0; j < k; j++) {
              int s_x = local_x + j * dilation;
              int s_y = local_y + i * dilation;
              int idx = s_y * tile_width + s_x;
              sum += s_input[idx] * s_weight[i * k + j];
          }
      }
      if (bias != nullptr) {
          sum += bias[c];
      }
      int out_idx = n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
      output[out_idx] = sum;
  }
}

// Pointwise convolution kernel leveraging shared memory for reusing the weight vector.
// Each block computes a tile of the output for one (batch, output channel).

template <typename scalar_t>
__global__ void shared_mem_pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,   // [batch, in_channels, h, w] (depthwise output)
    const scalar_t* __restrict__ weight,  // [out_channels, in_channels]
    const scalar_t* __restrict__ bias,    // [out_channels] or nullptr
    scalar_t* __restrict__ output,        // [batch, out_channels, h, w]
    int batch,
    int in_channels,
    int out_channels,
    int h, int w) {

  // Determine which image and which output channel this block handles
  int linear_idx = blockIdx.z;  // linear index over (batch, out_channel)
  int n = linear_idx / out_channels;
  int oc = linear_idx % out_channels;

  int block_out_x = blockIdx.x * blockDim.x;
  int block_out_y = blockIdx.y * blockDim.y;

  // Dynamic shared memory for the weight vector of this output channel (size = in_channels)
  extern __shared__ char smem[];
  scalar_t* s_weight = reinterpret_cast<scalar_t*>(smem);

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  // Load the entire weight row for this output channel into shared memory.
  for (int i = tid; i < in_channels; i += blockDim.x * blockDim.y) {
      s_weight[i] = weight[oc * in_channels + i];
  }
  __syncthreads();

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ow = block_out_x + tx;
  int oh = block_out_y + ty;

  if (ow < w && oh < h) {
      scalar_t sum = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);
      for (int ic = 0; ic < in_channels; ic++) {
          int inp_idx = n * in_channels * h * w + ic * h * w + oh * w + ow;
          sum += input[inp_idx] * s_weight[ic];
      }
      int out_idx = n * out_channels * h * w + oc * h * w + oh * w + ow;
      output[out_idx] = sum;
  }
}

// Core CUDA forward function that calls the shared memory kernels.

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

  int k = depthwise_weight.size(2);  // assuming depthwise_weight shape [channels, 1, k, k]
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  // Depthwise convolution output
  auto depthwise_output = torch::empty({batch, channels, out_h, out_w}, x.options());

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((out_w + block.x - 1) / block.x,
            (out_h + block.y - 1) / block.y,
            batch * channels);

  // Calculate shared memory size: input tile + filter weights.
  // Use x.element_size() to get the size of each element in bytes.
  int tile_width = block.x * stride + (k - stride);
  int tile_height = block.y * stride + (k - stride);
  size_t shared_mem_size = (tile_width * tile_height + k * k) * x.element_size();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "shared_mem_depthwise_conv2d_cuda", ([&] {
    shared_mem_depthwise_conv2d_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
        x.data_ptr<scalar_t>(),
        depthwise_weight.data_ptr<scalar_t>(),
        depthwise_bias.defined() && depthwise_bias.numel() > 0 ? depthwise_bias.data_ptr<scalar_t>() : nullptr,
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

  // Pointwise convolution
  int out_channels = pointwise_weight.size(0);
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

  dim3 block_pw(TILE_SIZE, TILE_SIZE);
  dim3 grid_pw((out_w + block_pw.x - 1) / block_pw.x,
               (out_h + block_pw.y - 1) / block_pw.y,
               batch * out_channels);

  // Shared memory size for pointwise: one weight vector of length 'channels'
  size_t shared_mem_size_pw = channels * x.element_size();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "shared_mem_pointwise_conv2d_cuda", ([&] {
    shared_mem_pointwise_conv2d_kernel<scalar_t><<<grid_pw, block_pw, shared_mem_size_pw>>>(
        depthwise_output.data_ptr<scalar_t>(),
        pointwise_weight.data_ptr<scalar_t>(),
        pointwise_bias.defined() && pointwise_bias.numel() > 0 ? pointwise_bias.data_ptr<scalar_t>() : nullptr,
        output.data_ptr<scalar_t>(),
        batch,
        channels,
        out_channels,
        out_h, out_w);
  }));

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Pointwise kernel launch error: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Helper: convert a py::object to an at::Tensor. Supports both direct tensors and objects with a 'data' attribute.
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

// Wrapper function exposed to Python.
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
  m.def("forward", &forward_wrapper, "CUDA depthwise separable convolution forward using shared memory");
}
