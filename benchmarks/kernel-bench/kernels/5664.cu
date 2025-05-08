#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Device function: Compute multi-dimensional indices (b, c, oh, ow) from flat index
template <typename scalar_t>
__device__ __forceinline__ void get_output_indices(
    const int index,
    const int output_width,
    const int output_height,
    const int channels,
    int &b, int &c, int &oh, int &ow) {
  ow = index % output_width;
  oh = (index / output_width) % output_height;
  c = (index / (output_width * output_height)) % channels;
  b = index / (output_width * output_height * channels);
}

// Device function: Compute the max pooling operation for one output element
template <typename scalar_t>
__device__ __forceinline__ scalar_t maxpool_window(
    const scalar_t* __restrict__ input,
    const int b, const int c, const int oh, const int ow,
    const int input_height, const int input_width,
    const int kernel_size, const int stride,
    const int padding, const int dilation,
    const int channels) {

  scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
  const int channel_area = input_height * input_width;
  // Precompute the offset for the (b, c) plane
  const int offset = b * channels * channel_area + c * channel_area;
  
  // Compute starting indices of the pooling window
  const int start_h = oh * stride - padding;
  const int start_w = ow * stride - padding;

  // Iterate over pooling window
  for (int kh = 0; kh < kernel_size; ++kh) {
    const int ih = start_h + kh * dilation;
    if (ih < 0 || ih >= input_height) continue;
    for (int kw = 0; kw < kernel_size; ++kw) {
      const int iw = start_w + kw * dilation;
      if (iw < 0 || iw >= input_width) continue;
      const int input_idx = offset + ih * input_width + iw;
      const scalar_t val = input[input_idx];
      if (val > max_val) {
        max_val = val;
      }
    }
  }
  return max_val;
}

// Modular MaxPool2D kernel using grid-stride loop and modular device functions
template <typename scalar_t>
__global__ void modular_fast_maxpool2d_kernel(
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
  const int gridStride = blockDim.x * gridDim.x;

  for (; idx < total; idx += gridStride) {
    int b, c, oh, ow;
    get_output_indices<scalar_t>(idx, output_width, output_height, channels, b, c, oh, ow);
    output[idx] = maxpool_window<scalar_t>(
        input, b, c, oh, ow,
        input_height, input_width,
        kernel_size, stride, padding, dilation,
        channels);
  }
}

// Host function to launch the kernel
torch::Tensor modular_fast_maxpool2d_cuda_forward(
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

  // Compute output dimensions as per pooling formula
  const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
  const int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

  auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
  
  const int total = batch_size * channels * output_height * output_width;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "modular_fast_maxpool2d_cuda_forward", ([&] {
    modular_fast_maxpool2d_kernel<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size, channels,
        input_height, input_width,
        output_height, output_width,
        kernel_size, stride, padding, dilation);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &modular_fast_maxpool2d_cuda_forward, "Modular Fast MaxPool2D forward (CUDA)");
}
