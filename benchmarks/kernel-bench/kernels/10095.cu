#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <cmath>

namespace py = pybind11;

#define TILE_SIZE 16
#define TILE_OC 8

// Optimized fused kernel for depthwise and pointwise convolution
// This kernel combines the operations into a single pass, reducing memory access overhead

template <typename scalar_t>
__global__ void optimized_fused_conv2d_kernel(
    const scalar_t* __restrict__ input,             // [batch, in_channels, in_h, in_w]
    const scalar_t* __restrict__ depthwise_weight,  // [in_channels, 1, k, k]
    const scalar_t* __restrict__ pointwise_weight,  // [out_channels, in_channels]
    const scalar_t* __restrict__ depthwise_bias,    // [in_channels] or nullptr
    const scalar_t* __restrict__ pointwise_bias,    // [out_channels] or nullptr
    scalar_t* __restrict__ output,                  // [batch, out_channels, out_h, out_w]
    int batch,
    int in_channels,
    int out_channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k,
    int stride,
    int padding,
    int dilation) {

  int n = blockIdx.z;  // one block layer per image in batch
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;

  if (ow < out_w && oh < out_h) {
    for (int oc_tile = 0; oc_tile < out_channels; oc_tile += TILE_OC) {
      scalar_t acc[TILE_OC] = {0};

      for (int c = 0; c < in_channels; c++) {
        scalar_t dw = 0;
        for (int i = 0; i < k; i++) {
          for (int j = 0; j < k; j++) {
            int ih = oh * stride - padding + i * dilation;
            int iw = ow * stride - padding + j * dilation;
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
              int input_idx = n * (in_channels * in_h * in_w) +
                              c * (in_h * in_w) +
                              ih * in_w + iw;
              int weight_dw_idx = c * (k * k) + i * k + j;
              dw += input[input_idx] * depthwise_weight[weight_dw_idx];
            }
          }
        }
        if (depthwise_bias != nullptr) {
          dw += depthwise_bias[c];
        }

        #pragma unroll
        for (int t = 0; t < TILE_OC; t++) {
          int oc = oc_tile + t;
          if (oc < out_channels) {
            int weight_pw_idx = oc * in_channels + c;
            acc[t] += dw * pointwise_weight[weight_pw_idx];
          }
        }
      }

      for (int t = 0; t < TILE_OC; t++) {
        int oc = oc_tile + t;
        if (oc < out_channels) {
          if (pointwise_bias != nullptr)
            acc[t] += pointwise_bias[oc];
          int output_idx = n * (out_channels * out_h * out_w) +
                           oc * (out_h * out_w) +
                           oh * out_w + ow;
          output[output_idx] = acc[t];
        }
      }
    }
  }
}

// Optimized forward function

torch::Tensor optimized_fused_forward_cuda(
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

  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE,
            (out_h + TILE_SIZE - 1) / TILE_SIZE,
            batch);

  const void* depthwise_bias_ptr = (depthwise_bias.defined() && depthwise_bias.numel() > 0) ? depthwise_bias.data_ptr() : nullptr;
  const void* pointwise_bias_ptr = (pointwise_bias.defined() && pointwise_bias.numel() > 0) ? pointwise_bias.data_ptr() : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optimized_fused_conv2d_cuda", ([&] {
    optimized_fused_conv2d_kernel<scalar_t><<<grid, block>>>(
        x.data_ptr<scalar_t>(),
        depthwise_weight.data_ptr<scalar_t>(),
        pointwise_weight.data_ptr<scalar_t>(),
        reinterpret_cast<const scalar_t*>(depthwise_bias_ptr),
        reinterpret_cast<const scalar_t*>(pointwise_bias_ptr),
        output.data_ptr<scalar_t>(),
        batch,
        in_channels,
        out_channels,
        in_h, in_w,
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

at::Tensor optimized_fused_forward_wrapper(py::object x_obj,
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

  return optimized_fused_forward_cuda(x, depthwise_weight, pointwise_weight,
                                      depthwise_bias, pointwise_bias,
                                      stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &optimized_fused_forward_wrapper, "Optimized fused CUDA depthwise separable convolution");
}
