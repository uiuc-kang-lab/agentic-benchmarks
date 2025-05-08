#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Device function for 2x2 max pooling
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_pool_max_2x2(
    const scalar_t* __restrict__ input,
    int base_index,
    int input_width,
    int input_height,
    int start_h,
    int start_w,
    int dilation) {
  scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
  #pragma unroll
  for (int kh = 0; kh < 2; ++kh) {
    int ih = start_h + kh * dilation;
    if (ih < 0 || ih >= input_height) continue;
    int row_index = base_index + ih * input_width;
    #pragma unroll
    for (int kw = 0; kw < 2; ++kw) {
      int iw = start_w + kw * dilation;
      if (iw < 0 || iw >= input_width) continue;
      max_val = max(max_val, __ldg(&input[row_index + iw]));
    }
  }
  return max_val;
}

// Device function for 3x3 max pooling
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_pool_max_3x3(
    const scalar_t* __restrict__ input,
    int base_index,
    int input_width,
    int input_height,
    int start_h,
    int start_w,
    int dilation) {
  scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
  #pragma unroll
  for (int kh = 0; kh < 3; ++kh) {
    int ih = start_h + kh * dilation;
    if (ih < 0 || ih >= input_height) continue;
    int row_index = base_index + ih * input_width;
    #pragma unroll
    for (int kw = 0; kw < 3; ++kw) {
      int iw = start_w + kw * dilation;
      if (iw < 0 || iw >= input_width) continue;
      max_val = max(max_val, __ldg(&input[row_index + iw]));
    }
  }
  return max_val;
}

// General device function for arbitrary kernel sizes
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_pool_max_general(
    const scalar_t* __restrict__ input,
    int base_index,
    int input_width,
    int input_height,
    int start_h,
    int start_w,
    int kernel_size,
    int dilation) {
  scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
  for (int kh = 0; kh < kernel_size; ++kh) {
    int ih = start_h + kh * dilation;
    if (ih < 0 || ih >= input_height) continue;
    int row_index = base_index + ih * input_width;
    for (int kw = 0; kw < kernel_size; ++kw) {
      int iw = start_w + kw * dilation;
      if (iw < 0 || iw >= input_width) continue;
      max_val = max(max_val, __ldg(&input[row_index + iw]));
    }
  }
  return max_val;
}

// Modular max pooling kernel that calls device helper functions
// Template parameter KERNEL_SIZE is used for compile-time unrolling if > 0; otherwise, runtime_kernel_size is used.

template <typename scalar_t, int KERNEL_SIZE>
__global__ void modular_max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int stride,
    int padding,
    int dilation,
    int runtime_kernel_size) {

  // Map each thread to one output element
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;
  int bc = blockIdx.z;  // combined batch and channel index

  if (ow >= output_width || oh >= output_height)
    return;

  int b = bc / channels;
  int c = bc % channels;

  int base_index = b * channels * input_height * input_width + c * input_height * input_width;
  int start_h = oh * stride - padding;
  int start_w = ow * stride - padding;

  scalar_t max_val;
  if constexpr (KERNEL_SIZE == 2) {
    max_val = compute_pool_max_2x2<scalar_t>(input, base_index, input_width, input_height, start_h, start_w, dilation);
  } else if constexpr (KERNEL_SIZE == 3) {
    max_val = compute_pool_max_3x3<scalar_t>(input, base_index, input_width, input_height, start_h, start_w, dilation);
  } else if constexpr (KERNEL_SIZE > 0) {
    max_val = compute_pool_max_general<scalar_t>(input, base_index, input_width, input_height, start_h, start_w, KERNEL_SIZE, dilation);
  } else {
    // Fallback to runtime kernel size
    max_val = compute_pool_max_general<scalar_t>(input, base_index, input_width, input_height, start_h, start_w, runtime_kernel_size, dilation);
  }

  int out_index = b * (channels * output_height * output_width) + 
                  c * (output_height * output_width) +
                  oh * output_width + ow;
  output[out_index] = max_val;
}

// Host launcher function

torch::Tensor modular_max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
  int batch_size = input.size(0);
  int channels = input.size(1);
  int input_height = input.size(2);
  int input_width = input.size(3);
  int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
  int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

  auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

  dim3 threads(16, 16);
  dim3 blocks((output_width + threads.x - 1) / threads.x,
              (output_height + threads.y - 1) / threads.y,
              batch_size * channels);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "modular_max_pool2d_cuda_forward", ([&] {
    if (kernel_size == 2) {
      modular_max_pool2d_kernel<scalar_t, 2><<<blocks, threads>>>(
          input.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          batch_size,
          channels,
          input_height,
          input_width,
          output_height,
          output_width,
          stride,
          padding,
          dilation,
          0); // unused
    } else if (kernel_size == 3) {
      modular_max_pool2d_kernel<scalar_t, 3><<<blocks, threads>>>(
          input.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          batch_size,
          channels,
          input_height,
          input_width,
          output_height,
          output_width,
          stride,
          padding,
          dilation,
          0);
    } else {
      modular_max_pool2d_kernel<scalar_t, -1><<<blocks, threads>>>(
          input.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          batch_size,
          channels,
          input_height,
          input_width,
          output_height,
          output_width,
          stride,
          padding,
          dilation,
          kernel_size);
    }
  }));
  
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &modular_max_pool2d_cuda_forward, "Modular Max Pool 2D forward (CUDA)");
}
