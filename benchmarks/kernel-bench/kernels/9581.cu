#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// This kernel assigns one warp per output element. It loads the kernel weights into shared memory
// (per-warp allocation) to reduce redundant global memory accesses and then uses warp-level
// primitives to reduce partial sums computed by threads of the warp over the convolution window.

template <typename scalar_t>
__global__ void depthwiseConv2DKernelOpt(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int total)
{
    // Each warp computes one output element.
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int warps_per_block = blockDim.x / 32;
    int global_warp_id = blockIdx.x * warps_per_block + warp_id;

    if (global_warp_id >= total) return;

    // Decompose global_warp_id into (n, c, h_out, w_out):
    int w_out = global_warp_id % out_width;
    int tmp = global_warp_id / out_width;
    int h_out = tmp % out_height;
    tmp /= out_height;
    int c = tmp % in_channels;
    int n = tmp / in_channels;

    // Use dynamic shared memory to cache kernel weights for this warp.
    // Each warp gets a segment of size kernel_size * kernel_size.
    extern __shared__ char smem[];
    scalar_t* shared_weight = reinterpret_cast<scalar_t*>(smem) + warp_id * (kernel_size * kernel_size);
    int kernel_elems = kernel_size * kernel_size;

    // Load the kernel weights for channel 'c' into shared memory. Only one thread per warp does this.
    if (lane == 0) {
      for (int i = 0; i < kernel_elems; i++) {
        shared_weight[i] = w[c * kernel_elems + i];
      }
    }
    __syncthreads();

    // Each thread in the warp processes a subset of the kernel positions.
    scalar_t sum = 0;
    for (int idx = lane; idx < kernel_elems; idx += 32) {
      int kh = idx / kernel_size;
      int kw = idx % kernel_size;
      int h_in = h_out * stride - padding + kh;
      int w_in = w_out * stride - padding + kw;
      if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
        int x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
        sum += x[x_index] * shared_weight[idx];
      }
    }

    // Warp-level reduction using __shfl_down_sync.
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(mask, sum, offset);
    }

    // Thread with lane 0 writes the result.
    if (lane == 0) {
      sum += b[c];
      int out_index = ((n * in_channels + c) * out_height + h_out) * out_width + w_out;
      out[out_index] = sum;
    }
}

// The forward implementation for the optimized depthwise conv2d kernel.

torch::Tensor forward_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int groups) {

  const int batch_size = x.size(0);
  const int in_channels = x.size(1);
  const int in_height = x.size(2);
  const int in_width = x.size(3);

  const int kernel_size = weight.size(2);  // expecting weight shape (in_channels, 1, K, K)
  const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
  const int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

  auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

  // Total output elements and each warp computes one element.
  const int total = batch_size * in_channels * out_height * out_width;

  // Choose block size: 256 threads per block yields 8 warps per block.
  const int threads = 256;
  const int warps_per_block = threads / 32;
  const int blocks = (total + warps_per_block - 1) / warps_per_block;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward_opt", ([&] {
    size_t shared_size = (threads / 32) * kernel_size * kernel_size * sizeof(scalar_t);
    depthwiseConv2DKernelOpt<scalar_t><<<blocks, threads, shared_size>>>(
        x.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        batch_size, in_channels, in_height, in_width,
        kernel_size, out_height, out_width,
        stride, padding, total);
  }));

  return out;
}

// Wrap forward_impl to handle the optional bias argument from Python.

torch::Tensor forward_wrap(
    torch::Tensor x,
    torch::Tensor weight,
    pybind11::object bias_obj,
    int stride,
    int padding,
    int groups) {
  torch::Tensor bias;
  if (bias_obj.is_none()) {
    bias = torch::zeros({x.size(1)}, x.options());
  } else {
    bias = bias_obj.cast<torch::Tensor>();
  }
  return forward_impl(x, weight, bias, stride, padding, groups);
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "forward",
      &forward_wrap,
      "Optimized Depthwise Conv2D forward using shared memory and warp-level reductions",
      py::arg("x"),
      py::arg("weight"),
      py::arg("bias") = py::none(),
      py::arg("stride") = 1,
      py::arg("padding") = 0,
      py::arg("groups") = 1
  );
}
