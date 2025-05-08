#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Optimized max pooling 2D kernel using a grid-stride loop and tunable block size
template <typename scalar_t>
__global__ void max_pool2d_optimized_kernel(
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
    const int dilation,
    const int total_elements) {

  // Grid-stride loop for better load balancing across blocks
  for (int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
       output_idx < total_elements;
       output_idx += blockDim.x * gridDim.x) {

    int ow = output_idx % output_width;
    int oh = (output_idx / output_width) % output_height;
    int c  = (output_idx / (output_width * output_height)) % channels;
    int b  = output_idx / (output_width * output_height * channels);

    // Initialize max value with -infinity
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Iterate over the kernel window
    for (int kh = 0; kh < kernel_size; kh++) {
      for (int kw = 0; kw < kernel_size; kw++) {
        int ih = oh * stride - padding + kh * dilation;
        int iw = ow * stride - padding + kw * dilation;
        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
          int input_idx = b * (channels * input_height * input_width) +
                          c * (input_height * input_width) +
                          ih * input_width +
                          iw;
          scalar_t val = input[input_idx];
          max_val = (val > max_val) ? val : max_val;
        }
      }
    }

    output[output_idx] = max_val;
  }
}

// Forward function that allows for block size tuning (e.g., 32, 64, 128, 256, 512)
torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int block_size = 256) {  // Default block size is 256; can be tuned

  const auto batch_size = input.size(0);
  const auto channels   = input.size(1);
  const auto input_height = input.size(2);
  const auto input_width  = input.size(3);

  const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
  const auto output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

  int total_elements = batch_size * channels * output_height * output_width;
  auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

  const int threads = block_size;
  const int blocks = (total_elements + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
    max_pool2d_optimized_kernel<scalar_t><<<blocks, threads>>>(
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
        dilation,
        total_elements);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA optimized)");
}
