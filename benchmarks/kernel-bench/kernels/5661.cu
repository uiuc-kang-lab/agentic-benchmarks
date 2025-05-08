#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// This kernel minimizes warp divergence by splitting the pooling operation
// into two cases: one for fully valid pooling windows (no boundary checks needed) and
// one for partial windows (with boundary checks). This reduces divergent branching
// within warps for the majority of the output (central regions).

template <typename scalar_t>
__global__ void warp_uniform_maxpool2d_kernel(
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
    // Compute the output indices
    int ow = idx % output_width;
    int oh = (idx / output_width) % output_height;
    int c  = (idx / (output_width * output_height)) % channels;
    int b  = idx / (output_width * output_height * channels);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    // Compute the top-left corner of the pooling window
    int start_h = oh * stride - padding;
    int start_w = ow * stride - padding;

    // Precompute pointer offset for the (b, c) plane in the input
    int offset = b * (channels * input_height * input_width) + c * (input_height * input_width);

    // Check if the pooling window is fully within bounds
    bool fully_valid = (start_h >= 0) && (start_w >= 0) &&
                       ((start_h + (kernel_size - 1) * dilation) < input_height) &&
                       ((start_w + (kernel_size - 1) * dilation) < input_width);

    if (fully_valid) {
      // No boundary checks inside these loops
      for (int kh = 0; kh < kernel_size; ++kh) {
        int ih = start_h + kh * dilation;
        int base = offset + ih * input_width;
        for (int kw = 0; kw < kernel_size; ++kw) {
          int iw = start_w + kw * dilation;
          scalar_t temp = input[base + iw];
          max_val = temp > max_val ? temp : max_val;
        }
      }
    } else {
      // For partial windows, perform boundary checks;
      // Although this branch has conditionals, such cases occur mainly at the borders.
      for (int kh = 0; kh < kernel_size; ++kh) {
        int ih = start_h + kh * dilation;
        if (ih < 0 || ih >= input_height) continue;
        int base = offset + ih * input_width;
        for (int kw = 0; kw < kernel_size; ++kw) {
          int iw = start_w + kw * dilation;
          if (iw < 0 || iw >= input_width) continue;
          scalar_t temp = input[base + iw];
          max_val = temp > max_val ? temp : max_val;
        }
      }
    }

    output[idx] = max_val;
  }
}

// Host-code that sets up kernel launch parameters and dispatches the CUDA kernel

torch::Tensor warp_uniform_maxpool2d_cuda_forward(
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

  // Compute output dimensions as per usual pooling formula
  const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
  const int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

  auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

  const int total = batch_size * channels * output_height * output_width;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "warp_uniform_maxpool2d_cuda_forward", ([&] {
    warp_uniform_maxpool2d_kernel<scalar_t><<<blocks, threads>>>(
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
  m.def("forward", &warp_uniform_maxpool2d_cuda_forward, "Warp Uniform Max Pool 2D forward (CUDA)");
}
