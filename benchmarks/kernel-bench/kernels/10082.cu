#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Block tile size for output
#define BLOCK_DIM 16

// Depthwise Convolution Kernel with Shared Memory for Coalesced Global Memory Access
// This kernel loads a tile of the input into shared memory to ensure that global memory accesses
// are coalesced when reading the input patch required for each output element.

template <typename scalar_t>
__global__ void depthwise_conv2d_coalesced_kernel(
    const scalar_t* __restrict__ input,    // [batch, channels, in_h, in_w]
    const scalar_t* __restrict__ weight,   // [channels, 1, k, k]
    const scalar_t* __restrict__ bias,     // [channels] or nullptr
    scalar_t* __restrict__ output,         // [batch, channels, out_h, out_w]
    int batch,
    int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k,
    int stride,
    int padding,
    int dilation) {

  // Each block computes a tile of the output for one channel of one batch image
  int bc = blockIdx.z;  // combined index for batch and channel
  int n = bc / channels;
  int c = bc % channels;

  // Top-left corner of the output tile computed by this block
  int out_tile_x = blockIdx.x * blockDim.x;
  int out_tile_y = blockIdx.y * blockDim.y;

  // Thread indices within the block
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Global output coordinates (in output feature map)
  int ow = out_tile_x + tx;
  int oh = out_tile_y + ty;

  // Calculate the dimensions required for the shared memory tile
  // The tile needs to cover the region of the input used to compute the block's output tile
  // For each output pixel, the receptive field is k x k with dilation. With stride, the horizontal
  // span of the tile is: blockDim.x * stride + (k - stride) (similar for vertical dimension).
  int sm_tile_w = blockDim.x * stride + (k - stride);
  int sm_tile_h = blockDim.y * stride + (k - stride);

  // Starting position in the input corresponding to the top-left output of this block
  int in_tile_x = out_tile_x * stride - padding;
  int in_tile_y = out_tile_y * stride - padding;

  // Allocate shared memory dynamically
  extern __shared__ char smem[];
  scalar_t* sm_input = reinterpret_cast<scalar_t*>(smem);

  // Load the required input tile into shared memory in a coalesced manner
  // Each thread loads multiple elements if necessary
  for (int i = ty; i < sm_tile_h; i += blockDim.y) {
    for (int j = tx; j < sm_tile_w; j += blockDim.x) {
      int in_x = in_tile_x + j;
      int in_y = in_tile_y + i;
      scalar_t val = 0;
      if (in_x >= 0 && in_x < in_w && in_y >= 0 && in_y < in_h) {
        int input_idx = n * channels * in_h * in_w + c * in_h * in_w + in_y * in_w + in_x;
        val = input[input_idx];
      }
      sm_input[i * sm_tile_w + j] = val;
    }
  }
  __syncthreads();

  // If the thread's corresponding output coordinate is within bounds, compute the convolution
  if (ow < out_w && oh < out_h) {
    scalar_t sum = 0;
    // Iterate over the convolution kernel window
    for (int i = 0; i < k; ++i) {
      for (int j = 0; j < k; ++j) {
        // Compute the position in the shared memory tile
        int sm_y = ty * stride + i * dilation;
        int sm_x = tx * stride + j * dilation;
        // No need to check bounds since shared memory tile was sized to cover the entire receptive field
        int weight_idx = c * k * k + i * k + j;
        sum += sm_input[sm_y * sm_tile_w + sm_x] * weight[weight_idx];
      }
    }
    if (bias != nullptr) {
      sum += bias[c];
    }
    int output_idx = n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
    output[output_idx] = sum;
  }
}

// Pointwise Convolution Kernel with coalesced global memory accesses
// For 1x1 convolution, each thread computes one output pixel. Threads along the x-dimension
// access consecutive memory locations in the input and output, ensuring coalescing.

template <typename scalar_t>
__global__ void pointwise_conv2d_coalesced_kernel(
    const scalar_t* __restrict__ input,    // [batch, in_channels, h, w] (depthwise output)
    const scalar_t* __restrict__ weight,   // [out_channels, in_channels]
    const scalar_t* __restrict__ bias,     // [out_channels] or nullptr
    scalar_t* __restrict__ output,         // [batch, out_channels, h, w]
    int batch,
    int in_channels,
    int out_channels,
    int h, int w) {

  // Each block computes a tile for one output channel of one batch image
  int boc = blockIdx.z; // combined index for batch and output channel
  int n = boc / out_channels;
  int oc = boc % out_channels;

  int out_tile_x = blockIdx.x * blockDim.x;
  int out_tile_y = blockIdx.y * blockDim.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ow = out_tile_x + tx;
  int oh = out_tile_y + ty;

  if (ow < w && oh < h) {
    scalar_t sum = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);
    // Sum over all input channels
    for (int ic = 0; ic < in_channels; ++ic) {
      int input_idx = n * in_channels * h * w + ic * h * w + oh * w + ow;
      int weight_idx = oc * in_channels + ic;
      sum += input[input_idx] * weight[weight_idx];
    }
    int output_idx = n * out_channels * h * w + oc * h * w + oh * w + ow;
    output[output_idx] = sum;
  }
}

// Core CUDA forward function that calls the coalesced kernels

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

  // Get dimensions of input tensor (N, C, H, W)
  int batch = x.size(0);
  int channels = x.size(1);
  int in_h = x.size(2);
  int in_w = x.size(3);

  // Depthwise weight shape: [channels, 1, k, k]
  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  // Output tensor for depthwise convolution
  auto depthwise_output = torch::empty({batch, channels, out_h, out_w}, x.options());

  // Configure grid and block for depthwise kernel
  dim3 block(BLOCK_DIM, BLOCK_DIM);
  dim3 grid((out_w + block.x - 1) / block.x,
            (out_h + block.y - 1) / block.y,
            batch * channels);

  // Compute shared memory size needed for depthwise kernel
  int sm_tile_w = block.x * stride + (k - stride);
  int sm_tile_h = block.y * stride + (k - stride);
  size_t shared_mem_size = sizeof(float) * sm_tile_w * sm_tile_h;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_coalesced_cuda", ([&] {
    depthwise_conv2d_coalesced_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
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

  // Pointwise convolution: weight shape is [out_channels, in_channels]
  int out_channels = pointwise_weight.size(0);
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

  dim3 block_pw(BLOCK_DIM, BLOCK_DIM);
  dim3 grid_pw((out_w + block_pw.x - 1) / block_pw.x,
               (out_h + block_pw.y - 1) / block_pw.y,
               batch * out_channels);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_coalesced_cuda", ([&] {
    pointwise_conv2d_coalesced_kernel<scalar_t><<<grid_pw, block_pw>>>(
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

// Helper: convert a py::object to an at::Tensor. Supports both raw tensors and objects with a 'data' attribute.

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
  m.def("forward", &forward_wrapper, "CUDA depthwise separable convolution forward with coalesced memory accesses");
}
