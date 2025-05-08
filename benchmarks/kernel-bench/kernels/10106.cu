#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define THREADS_PER_BLOCK 256
#define TILE_SIZE 16

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch,
    int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k,
    int stride,
    int padding,
    int dilation) {

  __shared__ scalar_t tile[TILE_SIZE][TILE_SIZE];
  
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * channels * out_h * out_w;
  if (index >= total)
      return;

  int ow = index % out_w;
  int tmp = index / out_w;
  int oh = tmp % out_h;
  tmp = tmp / out_h;
  int c = tmp % channels;
  int n = tmp / channels;

  scalar_t sum = 0;
  
  // Load tile into shared memory only if needed
  int tile_row = threadIdx.x / TILE_SIZE;
  int tile_col = threadIdx.x % TILE_SIZE;
  if (tile_row < TILE_SIZE && tile_col < TILE_SIZE) {
    int ih = oh * stride - padding + tile_row;
    int iw = ow * stride - padding + tile_col;
    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
      tile[tile_row][tile_col] = input[n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw];
    } else {
      tile[tile_row][tile_col] = 0;
    }
  }
  __syncthreads();

  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      int ih = oh * stride - padding + i * dilation;
      int iw = ow * stride - padding + j * dilation;
      if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
        if (i < TILE_SIZE && j < TILE_SIZE) {
          sum += tile[i][j] * weight[c * k * k + i * k + j];
        } else {
          int input_idx = n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
          sum += input[input_idx] * weight[c * k * k + i * k + j];
        }
      }
    }
  }
  
  if (bias != nullptr)
    sum += bias[c];
  output[index] = sum;
}

template <typename scalar_t>
__global__ void pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int h,
    int w) {

  __shared__ scalar_t shared_input[TILE_SIZE];
  
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * out_channels * h * w;
  if (index >= total)
      return;

  int ow = index % w;
  int tmp = index / w;
  int oh = tmp % h;
  tmp = tmp / h;
  int oc = tmp % out_channels;
  int n = tmp / out_channels;

  scalar_t sum = 0;
  
  for (int ic_block = 0; ic_block < in_channels; ic_block += TILE_SIZE) {
    int remaining = min(TILE_SIZE, in_channels - ic_block);
    
    if (threadIdx.x < remaining) {
      int ic = ic_block + threadIdx.x;
      shared_input[threadIdx.x] = input[n * in_channels * h * w + ic * h * w + oh * w + ow];
    }
    __syncthreads();
    
    for (int i = 0; i < remaining; ++i) {
      sum += shared_input[i] * weight[oc * in_channels + (ic_block + i)];
    }
    __syncthreads();
  }
  
  if (bias != nullptr)
    sum += bias[oc];
  output[index] = sum;
}

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

  int batch = x.size(0);
  int in_channels = x.size(1);
  int in_h = x.size(2);
  int in_w = x.size(3);
  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());
  int total_depthwise = batch * in_channels * out_h * out_w;
  int threads = THREADS_PER_BLOCK;
  int blocks = (total_depthwise + threads - 1) / threads;

  const void* depthwise_bias_ptr = (depthwise_bias.defined() && depthwise_bias.numel() > 0)
                                     ? depthwise_bias.data_ptr()
                                     : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda", ([&] {
    depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
        x.data_ptr<scalar_t>(),
        depthwise_weight.data_ptr<scalar_t>(),
        reinterpret_cast<const scalar_t*>(depthwise_bias_ptr),
        depthwise_output.data_ptr<scalar_t>(),
        batch, in_channels, in_h, in_w, out_h, out_w,
        k, stride, padding, dilation);
  }));

  int out_channels = pointwise_weight.size(0);
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());
  int total_pointwise = batch * out_channels * out_h * out_w;
  blocks = (total_pointwise + threads - 1) / threads;

  const void* pointwise_bias_ptr = (pointwise_bias.defined() && pointwise_bias.numel() > 0)
                                     ? pointwise_bias.data_ptr()
                                     : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
    pointwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
        depthwise_output.data_ptr<scalar_t>(),
        pointwise_weight.data_ptr<scalar_t>(),
        reinterpret_cast<const scalar_t*>(pointwise_bias_ptr),
        output.data_ptr<scalar_t>(),
        batch, in_channels, out_channels, out_h, out_w);
  }));

  return output;
}

at::Tensor toTensor(const py::object& obj) {
  if (obj.is_none()) return at::Tensor();
  try {
    return obj.cast<at::Tensor>();
  } catch (const py::cast_error& e) {
    if (py::hasattr(obj, "data")) {
      return obj.attr("data").cast<at::Tensor>();
    }
    throw std::runtime_error("Expected a torch Tensor or Parameter.");
  }
}

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