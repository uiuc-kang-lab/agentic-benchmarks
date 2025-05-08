#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// CUDA kernel using shared memory and exploring optimal block size configurations
__global__ void conv_transpose2d_forward_kernel_optimal_blocks(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation) {
  // Using blockDim.x * blockDim.y to determine thread position
  int out_w = blockIdx.x * blockDim.x + threadIdx.x;
  int out_h = blockIdx.y * blockDim.y + threadIdx.y;

  int bo_idx = blockIdx.z;
  int o = bo_idx % out_channels;
  int b = bo_idx / out_channels;

  extern __shared__ float shared_weight[]; // size: in_channels * kernel_size * kernel_size

  // Cooperative loading of weights into shared memory
  int weight_count = in_channels * kernel_size * kernel_size;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = tid; i < weight_count; i += blockDim.x * blockDim.y) {
    int tmp = i;
    int q = tmp % kernel_size;
    tmp /= kernel_size;
    int p = tmp % kernel_size;
    int c = tmp / kernel_size;
    int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
    shared_weight[i] = weight[weight_idx];
  }
  __syncthreads();

  if (out_w >= out_width || out_h >= out_height)
    return;

  float out_val = bias[o];

  for (int c = 0; c < in_channels; ++c) {
    for (int p = 0; p < kernel_size; ++p) {
      int h_unscaled = out_h + padding - p * dilation;
      if (h_unscaled % stride != 0)
        continue;
      int h_in = h_unscaled / stride;
      if (h_in < 0 || h_in >= in_height)
        continue;
      for (int q = 0; q < kernel_size; ++q) {
        int w_unscaled = out_w + padding - q * dilation;
        if (w_unscaled % stride != 0)
          continue;
        int w_in = w_unscaled / stride;
        if (w_in < 0 || w_in >= in_width)
          continue;
        int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
        int weight_shared_idx = c * (kernel_size * kernel_size) + p * kernel_size + q;
        out_val += input[input_idx] * shared_weight[weight_shared_idx];
      }
    }
  }

  int output_idx = ((b * out_channels + o) * out_height + out_h) * out_width + out_w;
  output[output_idx] = out_val;
}

// CUDA launcher function with different block sizes
torch::Tensor conv_transpose2d_forward_cuda_optimal_blocks(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {

  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int in_height = input.size(2);
  int in_width = input.size(3);

  int out_channels = weight.size(1);
  int kernel_size = weight.size(2);

  int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

  // Trial block configurations for exploration
  // Configure threads using a variety of potential block dimensions
  // Typical choices can be 256 or 512 which are divisible by the warp size of 32
  const int block_sizes[] = {32, 64, 128, 256, 512};
  int best_block_size = 256; // default, but can test other block sizes to find optimal
  float best_time = FLT_MAX;

  for (int block_size : block_sizes) {
    dim3 block(block_size, 8);
    dim3 grid((out_width + block.x - 1) / block.x,
              (out_height + block.y - 1) / block.y,
              batch_size * out_channels);

    int weight_count = in_channels * kernel_size * kernel_size;
    size_t shared_mem_size = weight_count * sizeof(float);

    // Launch kernel to profile execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    conv_transpose2d_forward_kernel_optimal_blocks<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_size,
        out_height,
        out_width,
        stride,
        padding,
        dilation);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (milliseconds < best_time) {
      best_time = milliseconds;
      best_block_size = block_size;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  // Launch using the best block size determined
  dim3 block(best_block_size, 8);
  dim3 grid((out_width + block.x - 1) / block.x,
            (out_height + block.y - 1) / block.y,
            batch_size * out_channels);

  int weight_count = in_channels * kernel_size * kernel_size;
  size_t shared_mem_size = weight_count * sizeof(float);

  conv_transpose2d_forward_kernel_optimal_blocks<<<grid, block, shared_mem_size>>>(
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.data_ptr<float>(),
      output.data_ptr<float>(),
      batch_size,
      in_channels,
      out_channels,
      in_height,
      in_width,
      kernel_size,
      out_height,
      out_width,
      stride,
      padding,
      dilation);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in conv_transpose2d_forward_kernel_optimal_blocks: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Wrapper function
torch::Tensor conv_transpose2d_forward_wrapper_optimal_blocks(
    torch::Tensor input,
    torch::Tensor weight,
    pybind11::object bias_obj,
    int stride,
    int padding,
    int dilation) {

  int out_channels = weight.size(1);
  torch::Tensor bias;
  if (bias_obj.is(pybind11::none())) {
    bias = torch::zeros({out_channels}, weight.options());
  } else {
    bias = bias_obj.cast<torch::Tensor>();
  }

  return conv_transpose2d_forward_cuda_optimal_blocks(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper_optimal_blocks,
        "ConvTranspose2d forward with optimized block size (CUDA)",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}