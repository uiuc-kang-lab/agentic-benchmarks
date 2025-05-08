#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Define tile dimensions for output in the depthwise convolution kernel.
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// Depthwise convolution kernel using shared memory tiling.
// Each block processes one tile of the output for a specific (n, c) pair.

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel_shared(
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

  // Each block in z-dimension corresponds to a unique (n, c) pair.
  int nc = blockIdx.z;  // index in [0, batch * channels)
  int n = nc / channels;
  int c = nc % channels;

  // Determine the top-left corner of the output tile processed by this block.
  int out_tile_x = blockIdx.x * TILE_WIDTH;
  int out_tile_y = blockIdx.y * TILE_HEIGHT;

  // Thread's coordinate in the output tile.
  int tx = threadIdx.x;  // in [0, TILE_WIDTH)
  int ty = threadIdx.y;  // in [0, TILE_HEIGHT)

  // Calculate shared memory tile dimensions needed to cover the receptive field.
  // For output tile of size (TILE_HEIGHT x TILE_WIDTH), the required input patch has size:
  // sm_width = TILE_WIDTH * stride + (k - 1) * dilation
  // sm_height = TILE_HEIGHT * stride + (k - 1) * dilation
  int sm_width = TILE_WIDTH * stride + (k - 1) * dilation;
  int sm_height = TILE_HEIGHT * stride + (k - 1) * dilation;

  // Compute the top-left coordinate in the input corresponding to the output tile.
  int in_tile_x = out_tile_x * stride - padding;
  int in_tile_y = out_tile_y * stride - padding;

  // Allocate shared memory for the input tile
  extern __shared__ char shared_mem[];
  scalar_t* shmem = reinterpret_cast<scalar_t*>(shared_mem);

  // Total elements in shared memory tile.
  int sm_size = sm_width * sm_height;
  int threadId = threadIdx.y * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * blockDim.y;

  // Each thread cooperatively loads elements from global memory into shared memory.
  for (int idx = threadId; idx < sm_size; idx += num_threads) {
    int sh_y = idx / sm_width;
    int sh_x = idx % sm_width;
    int in_y = in_tile_y + sh_y;
    int in_x = in_tile_x + sh_x;
    scalar_t val = 0;
    if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
      int input_idx = n * channels * in_h * in_w + c * in_h * in_w + in_y * in_w + in_x;
      val = input[input_idx];
    }
    shmem[idx] = val;
  }

  // Synchronize threads to ensure the shared memory tile is fully loaded.
  __syncthreads();

  // Compute the output coordinate for this thread.
  int out_x = out_tile_x + tx;
  int out_y = out_tile_y + ty;
  if (out_x < out_w && out_y < out_h) {
    scalar_t sum = 0;
    // The top-left of the receptive field in shared memory for this output element
    // is at (ty * stride, tx * stride).
    int sh_y_base = ty * stride;
    int sh_x_base = tx * stride;
    
    // Perform convolution over the k x k kernel using shared memory.
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < k; j++) {
        int sh_y = sh_y_base + i * dilation;
        int sh_x = sh_x_base + j * dilation;
        int sm_index = sh_y * sm_width + sh_x;
        sum += shmem[sm_index] * weight[c * k * k + i * k + j];
      }
    }
    if (bias != nullptr)
      sum += bias[c];
    int out_index = n * channels * out_h * out_w + c * out_h * out_w + out_y * out_w + out_x;
    output[out_index] = sum;
  }
}

// Pointwise (1x1) convolution kernel using a grid-stride loop without shared memory.
template <typename scalar_t>
__global__ void pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,   // [batch, in_channels, h, w]
    const scalar_t* __restrict__ weight,  // [out_channels, in_channels]
    const scalar_t* __restrict__ bias,    // [out_channels] or nullptr
    scalar_t* __restrict__ output,        // [batch, out_channels, h, w]
    int batch,
    int in_channels,
    int out_channels,
    int h,
    int w) {
  int total = batch * out_channels * h * w;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
    int ow = i % w;
    int tmp = i / w;
    int oh = tmp % h;
    tmp = tmp / h;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;
    scalar_t sum = 0;
    for (int ic = 0; ic < in_channels; ic++) {
      int input_idx = n * in_channels * h * w + ic * h * w + oh * w + ow;
      int weight_idx = oc * in_channels + ic;
      sum += input[input_idx] * weight[weight_idx];
    }
    if (bias != nullptr)
      sum += bias[oc];
    output[i] = sum;
  }
}

// Core CUDA forward function that launches the depthwise (with shared memory) and pointwise kernels.

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

  int batch = x.size(0);
  int channels = x.size(1);
  int in_h = x.size(2);
  int in_w = x.size(3);
  int k = depthwise_weight.size(2); // assuming square kernel
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  // Launch the depthwise convolution kernel with shared memory tiling.
  // Grid dimensions (x, y) cover the output tiled by TILE_WIDTH x TILE_HEIGHT.
  // The z-dimension covers each (n, c) pair: gridDim.z = batch * channels.
  int grid_dim_x = (out_w + TILE_WIDTH - 1) / TILE_WIDTH;
  int grid_dim_y = (out_h + TILE_HEIGHT - 1) / TILE_HEIGHT;
  int grid_dim_z = batch * channels;
  dim3 depth_grid(grid_dim_x, grid_dim_y, grid_dim_z);
  dim3 depth_block(TILE_WIDTH, TILE_HEIGHT);

  // Calculate the required shared memory size
  int sm_width = TILE_WIDTH * stride + (k - 1) * dilation;
  int sm_height = TILE_HEIGHT * stride + (k - 1) * dilation;
  size_t shared_mem_size = sm_width * sm_height * sizeof(float);

  auto depthwise_output = torch::empty({batch, channels, out_h, out_w}, x.options());
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda_shared", ([&] {
    depthwise_conv2d_kernel_shared<scalar_t><<<depth_grid, depth_block, shared_mem_size>>>(
        x.data_ptr<scalar_t>(),
        depthwise_weight.data_ptr<scalar_t>(),
        depthwise_bias.defined() && depthwise_bias.numel() > 0 ? depthwise_bias.data_ptr<scalar_t>() : nullptr,
        depthwise_output.data_ptr<scalar_t>(),
        batch, channels,
        in_h, in_w,
        out_h, out_w,
        k, stride, padding, dilation);
  }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Depthwise kernel launch error: %s\n", cudaGetErrorString(err));
  }

  // Launch the pointwise convolution kernel (1x1 convolution)
  int out_channels = pointwise_weight.size(0);
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());
  int total_pointwise = batch * out_channels * out_h * out_w;
  int threads = 256;
  int blocks = (total_pointwise + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
    pointwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
        depthwise_output.data_ptr<scalar_t>(),
        pointwise_weight.data_ptr<scalar_t>(),
        pointwise_bias.defined() && pointwise_bias.numel() > 0 ? pointwise_bias.data_ptr<scalar_t>() : nullptr,
        output.data_ptr<scalar_t>(),
        batch, channels,
        out_channels,
        out_h, out_w);
  }));
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Pointwise kernel launch error: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Helper function to convert a py::object to an at::Tensor
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
  m.def("forward", &forward_wrapper, "CUDA depthwise separable convolution forward");
}
