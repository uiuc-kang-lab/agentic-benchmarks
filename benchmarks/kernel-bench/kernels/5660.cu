#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Using constant memory for kernel parameters that do not change during kernel execution
__constant__ int kernel_size_c;
__constant__ int stride_c;
__constant__ int padding_c;
__constant__ int dilation_c;

// Device function that computes max pooling for a given output element
__device__ __forceinline__ float compute_maxpool(const float* __restrict__ input,
                                                  int input_height, int input_width, int channels,
                                                  int b, int c, int oh, int ow) {
  
  float max_val = -std::numeric_limits<float>::infinity();
  // Pre-compute the offset for the (b, c) plane
  int input_plane = input_height * input_width;
  int offset = b * channels * input_plane + c * input_plane;

  // Iterate over the pooling window
  for (int kh = 0; kh < kernel_size_c; ++kh) {
    for (int kw = 0; kw < kernel_size_c; ++kw) {
      int ih = oh * stride_c - padding_c + kh * dilation_c;
      int iw = ow * stride_c - padding_c + kw * dilation_c;
      if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
        int idx = offset + ih * input_width + iw;
        float val = input[idx];
        if (val > max_val) {
          max_val = val;
        }
      }
    }
  }
  return max_val;
}

// Optimized kernel using constant memory
__global__ void constant_memory_maxpool2d_kernel(const float* __restrict__ input,
                                                  float* __restrict__ output,
                                                  int batch_size, int channels,
                                                  int input_height, int input_width,
                                                  int output_height, int output_width) {

  const int total = batch_size * channels * output_height * output_width;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gridSize = blockDim.x * gridDim.x;
  
  // Process output elements in a grid-stride loop
  for (; idx < total; idx += gridSize) {
    int ow = idx % output_width;
    int oh = (idx / output_width) % output_height;
    int c  = (idx / (output_width * output_height)) % channels;
    int b  = idx / (output_width * output_height * channels);

    output[idx] = compute_maxpool(input, input_height, input_width, channels, b, c, oh, ow);
  }
}

// Host function to launch the kernel using constant memory for kernel parameters
void launch_constant_memory_maxpool2d_kernel(
    torch::Tensor input,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

  // Copy the constants to constant memory
  cudaMemcpyToSymbol(kernel_size_c, &kernel_size, sizeof(int));
  cudaMemcpyToSymbol(stride_c, &stride, sizeof(int));
  cudaMemcpyToSymbol(padding_c, &padding, sizeof(int));
  cudaMemcpyToSymbol(dilation_c, &dilation, sizeof(int));

  const int batch_size = input.size(0);
  const int channels = input.size(1);
  const int input_height = input.size(2);
  const int input_width = input.size(3);

  const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
  const int output_width  = ((input_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

  const int total_elements = batch_size * channels * output_height * output_width;
  const int threads = 256;
  const int blocks = (total_elements + threads - 1) / threads;

  constant_memory_maxpool2d_kernel<<<blocks, threads>>>(
    input.data_ptr<float>(),
    output.data_ptr<float>(),
    batch_size, channels,
    input_height, input_width,
    output_height, output_width
  );

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", [](
      torch::Tensor input,
      int kernel_size,
      int stride,
      int padding,
      int dilation) {
    auto output = torch::empty(
        {input.size(0), input.size(1),
         (input.size(2) + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1,
         (input.size(3) + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1},
        input.options());
    launch_constant_memory_maxpool2d_kernel(input, output, kernel_size, stride, padding, dilation);
    return output;
  }, "Constant Memory Max Pool 2D forward (CUDA)");
}