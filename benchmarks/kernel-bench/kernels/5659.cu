#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Device function to compute the max pooling result for one output element
// Using __restrict__ to optimize memory access
// Use of __forceinline__ for potential performance improvement
// Precompute input_plane and offset to reduce redundant calculations
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_maxpool(
    const scalar_t* __restrict__ input,
    int b, int c, int oh, int ow,
    int input_height, int input_width,
    int kernel_size, int stride, int padding, int dilation,
    int channels) {

  scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
  int input_plane = input_height * input_width;
  int offset = b * channels * input_plane + c * input_plane;

  for (int kh = 0; kh < kernel_size; ++kh) {
    for (int kw = 0; kw < kernel_size; ++kw) {
      int ih = oh * stride - padding + kh * dilation;
      int iw = ow * stride - padding + kw * dilation;
      if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
        int in_idx = offset + ih * input_width + iw;
        scalar_t val = input[in_idx];
        if (val > max_val) {
          max_val = val;
        }
      }
    }
  }
  return max_val;
}

// Kernel that uses a grid-stride loop and the modular device function
// Incorporates error checking directly after kernel call
// Uses grid-stride loop for better GPU utilization
// __restrict__ pointers for improved memory access patterns
// Error checking moved directly after kernel launch
template <typename scalar_t>
__global__ void optimized_maxpool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {

  const int total = batch_size * channels * output_height * output_width;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int gridSize = blockDim.x * gridDim.x;

  for (; idx < total; idx += gridSize) {
    int ow = idx % output_width;
    int oh = (idx / output_width) % output_height;
    int c  = (idx / (output_width * output_height)) % channels;
    int b  = idx / (output_width * output_height * channels);

    scalar_t max_val = compute_maxpool(input, b, c, oh, ow,
                                         input_height, input_width,
                                         kernel_size, stride, padding, dilation,
                                         channels);
    output[idx] = max_val;
  }
}

// Host function to prepare parameters and launch the kernel
// Error checking moved directly after kernel launch
// Use of AT_DISPATCH_FLOATING_TYPES for flexibility
// Ensures the input tensor is CUDA and prepares output tensor
// Calculates output dimensions and launches optimized kernel
// Improved error handling

torch::Tensor optimized_modular_maxpool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");

  const int batch_size = input.size(0);
  const int channels = input.size(1);
  const int input_height = input.size(2);
  const int input_width = input.size(3);

  int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
  int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

  auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

  int total_elements = batch_size * channels * output_height * output_width;
  const int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_modular_maxpool2d_cuda_forward", ([&] {
    optimized_maxpool2d_kernel<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        dilation);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &optimized_modular_maxpool2d_cuda_forward, "Optimized Modular Max Pool 2D forward (CUDA)");
}
