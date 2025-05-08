#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define BLOCK_SIZE 256

// Optimized fused depthwise separable convolution kernel with shared memory reduction and minimal atomic operations.
template <typename scalar_t>
__global__ void optimized_fused_conv_kernel(
    const scalar_t* __restrict__ input,           // [batch, in_channels, in_h, in_w]
    const scalar_t* __restrict__ depthwise_weight,  // [in_channels, 1, k, k] stored as [in_channels * k * k]
    const scalar_t* __restrict__ depthwise_bias,    // [in_channels] or nullptr
    const scalar_t* __restrict__ pointwise_weight,  // [out_channels, in_channels] stored row-major
    const scalar_t* __restrict__ pointwise_bias,    // [out_channels] or nullptr
    scalar_t* __restrict__ output,                  // [batch, out_channels, out_h, out_w]
    int batch,
    int in_channels,
    int in_h, int in_w,
    int out_channels,
    int out_h, int out_w,
    int k,            // kernel size (assumed square kernel k x k)
    int stride,
    int padding,
    int dilation) {

  int out_idx = blockIdx.x;
  int total_outputs = batch * out_channels * out_h * out_w;
  if (out_idx >= total_outputs) return;

  int ow = out_idx % out_w;
  int tmp = out_idx / out_w;
  int oh = tmp % out_h;
  tmp = tmp / out_h;
  int oc = tmp % out_channels;
  int n = tmp / out_channels;

  int block_ic_offset = blockIdx.y * blockDim.x;
  int threadId = threadIdx.x;
  int ic_stride = gridDim.y * blockDim.x;

  extern __shared__ scalar_t shared_data[];
  scalar_t* partial_sums = shared_data;

  scalar_t local_sum = 0;
  for (int ic = block_ic_offset + threadId; ic < in_channels; ic += ic_stride) {
    scalar_t d_sum = 0;
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < k; j++) {
        int ih = oh * stride - padding + i * dilation;
        int iw = ow * stride - padding + j * dilation;
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
          int input_idx = n * (in_channels * in_h * in_w) + ic * (in_h * in_w) + ih * in_w + iw;
          int weight_idx = ic * (k * k) + i * k + j;
          d_sum += input[input_idx] * depthwise_weight[weight_idx];
        }
      }
    }
    if (depthwise_bias != nullptr) {
      d_sum += depthwise_bias[ic];
    }
    int pw_idx = oc * in_channels + ic;
    local_sum += d_sum * pointwise_weight[pw_idx];
  }

  partial_sums[threadId] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadId < s) {
      partial_sums[threadId] += partial_sums[threadId + s];
    }
    __syncthreads();
  }

  if (threadId == 0) {
    atomicAdd(&output[out_idx], partial_sums[0]);
    if (blockIdx.y == 0 && pointwise_bias != nullptr) {
      atomicAdd(&output[out_idx], pointwise_bias[oc]);
    }
  }
}

// Forward function
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
  int in_channels = x.size(1);
  int in_h = x.size(2);
  int in_w = x.size(3);

  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  int out_channels = pointwise_weight.size(0);

  auto output = torch::zeros({batch, out_channels, out_h, out_w}, x.options());

  int total_outputs = batch * out_channels * out_h * out_w;

  int grid_y = (in_channels + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 grid(total_outputs, grid_y);
  dim3 block(BLOCK_SIZE);

  size_t shared_memory_size = BLOCK_SIZE * sizeof(scalar_t);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optimized_fused_conv_cuda", ([&] {
    optimized_fused_conv_kernel<scalar_t><<<grid, block, shared_memory_size>>>(
        x.data_ptr<scalar_t>(),
        depthwise_weight.data_ptr<scalar_t>(),
        depthwise_bias.defined() && depthwise_bias.numel() > 0 ? depthwise_bias.data_ptr<scalar_t>() : nullptr,
        pointwise_weight.data_ptr<scalar_t>(),
        pointwise_bias.defined() && pointwise_bias.numel() > 0 ? pointwise_bias.data_ptr<scalar_t>() : nullptr,
        output.data_ptr<scalar_t>(),
        batch,
        in_channels,
        in_h, in_w,
        out_channels,
        out_h, out_w,
        k,
        stride,
        padding,
        dilation);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Optimized fused kernel launch error: %s\n", cudaGetErrorString(err));
  }
  return output;
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
  m.def("forward", &forward_wrapper, "Optimized fused CUDA depthwise separable convolution");
}