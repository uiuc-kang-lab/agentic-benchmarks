#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Optimized CUDA kernel for conv_transpose2d forward using shared memory reduction
// and warp-level primitives (__shfl_down_sync) for final stages. Each block computes
// one output element in parallel, where all threads in a block collaboratively accumulate
// the convolution sum over input channels and kernel elements.

__global__ void conv_transpose2d_forward_kernel_opt(
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

  // Each block computes one output pixel
  int out_idx = blockIdx.x;
  int total_outputs = batch_size * out_channels * out_height * out_width;
  if (out_idx >= total_outputs) return;

  // Decode out_idx into (b, o, h_out, w_out)
  int w_out = out_idx % out_width;
  int temp = out_idx / out_width;
  int h_out = temp % out_height;
  temp /= out_height;
  int o = temp % out_channels;
  int b = temp / out_channels;

  int total_calc = in_channels * kernel_size * kernel_size;
  float sum = 0.0f;

  // Allocate shared memory for reduction
  extern __shared__ float sdata[];

  // Each thread in the block accumulates over a subset of (c, p, q) indices
  for (int i = threadIdx.x; i < total_calc; i += blockDim.x) {
      int c = i / (kernel_size * kernel_size);
      int rem = i % (kernel_size * kernel_size);
      int p = rem / kernel_size;
      int q = rem % kernel_size;

      int h_unscaled = h_out + padding - p * dilation;
      if (h_unscaled % stride != 0) continue;
      int h_in = h_unscaled / stride;
      if (h_in < 0 || h_in >= in_height) continue;

      int w_unscaled = w_out + padding - q * dilation;
      if (w_unscaled % stride != 0) continue;
      int w_in = w_unscaled / stride;
      if (w_in < 0 || w_in >= in_width) continue;

      int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
      int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
      sum += input[input_idx] * weight[weight_idx];
  }

  // Store each thread's partial sum in shared memory
  sdata[threadIdx.x] = sum;
  __syncthreads();

  // Intra-block reduction in shared memory (binary tree reduction)
  for (int s = blockDim.x / 2; s > 32; s /= 2) {
      if (threadIdx.x < s) {
          sdata[threadIdx.x] += sdata[threadIdx.x + s];
      }
      __syncthreads();
  }

  // Final warp-level reduction using __shfl_down_sync
  if (threadIdx.x < 32) {
      volatile float* smem = sdata;
      for (int offset = 16; offset > 0; offset /= 2) {
          smem[threadIdx.x] += __shfl_down_sync(0xffffffff, smem[threadIdx.x], offset);
      }
  }

  // Thread 0 writes the final output (adding bias)
  if (threadIdx.x == 0) {
      int output_idx = ((b * out_channels + o) * out_height + h_out) * out_width + w_out;
      output[output_idx] = bias[o] + sdata[0];
  }
}


// CUDA launch wrapper for the optimized kernel
// Each block is assigned to one output pixel and uses a fixed number of threads.

torch::Tensor conv_transpose2d_forward_cuda_opt(
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
  int kernel_size = weight.size(2); // square kernel assumed

  int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

  int total_outputs = batch_size * out_channels * out_height * out_width;
  int threads = 256;  // fixed threads per block
  int blocks = total_outputs;  // one block per output element
  int shared_mem_size = threads * sizeof(float);

  conv_transpose2d_forward_kernel_opt<<<blocks, threads, shared_mem_size>>>(
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
    printf("Error in conv_transpose2d_forward_kernel_opt: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Wrapper to handle the possibility that bias is None

torch::Tensor conv_transpose2d_forward_wrapper(
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

  return conv_transpose2d_forward_cuda_opt(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper,
        "ConvTranspose2d forward (CUDA) optimized with shared memory and warp-level reduction",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}
