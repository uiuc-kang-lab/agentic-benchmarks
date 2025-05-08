#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Optimized CUDA kernel for 2D transposed convolution with branchless inner loops
// that use uniform control flow to minimize warp divergence. Instead of precomputing
// valid kernel indices into temporary arrays, the kernel iterates over the entire
// kernel extent and uses branchless (ternary) operators to compute a validity mask.

__global__ void conv_transpose2d_forward_kernel_branchless(
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

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * out_channels * out_height * out_width;
  if (index >= total)
    return;

  // Decode the flat index into (b, o, out_h, out_w)
  int w_out = index % out_width;
  int temp = index / out_width;
  int h_out = temp % out_height;
  temp /= out_height;
  int o = temp % out_channels;
  int b = temp / out_channels;

  // Compute base coordinates for the output location
  int base_h = h_out + padding;
  int base_w = w_out + padding;

  // Start with the bias
  float out_val = __ldg(&bias[o]);

  // Loop over input channels
  for (int c = 0; c < in_channels; c++) {
    // Loop over kernel height (p) and kernel width (q) uniformly
    for (int p = 0; p < kernel_size; p++) {
      int p_dilated = p * dilation;
      int h_unscaled = base_h - p_dilated;
      // Compute branchless validity for h dimension
      int valid_h = ((h_unscaled >= 0) && ((h_unscaled % stride) == 0) && ((h_unscaled / stride) < in_height)) ? 1 : 0;
      int h_in = valid_h ? (h_unscaled / stride) : 0;  // safe value if not valid (will be multiplied by 0)

      for (int q = 0; q < kernel_size; q++) {
        int q_dilated = q * dilation;
        int w_unscaled = base_w - q_dilated;
        // Compute branchless validity for w dimension
        int valid_w = ((w_unscaled >= 0) && ((w_unscaled % stride) == 0) && ((w_unscaled / stride) < in_width)) ? 1 : 0;
        int w_in = valid_w ? (w_unscaled / stride) : 0;
        int valid = valid_h * valid_w;  // 1 if both dimensions are valid, else 0

        // Use the ternary operator to load input value only if valid; otherwise, use 0.0f
        float input_val = valid ? __ldg(&input[((b * in_channels + c) * in_height + h_in) * in_width + w_in]) : 0.0f;
        float weight_val = __ldg(&weight[(((c * out_channels + o) * kernel_size + p) * kernel_size) + q]);
        out_val += input_val * weight_val;
      }
    }
  }

  int output_idx = ((b * out_channels + o) * out_height + h_out) * out_width + w_out;
  output[output_idx] = out_val;
}

// CUDA forward function for the branchless kernel
torch::Tensor conv_transpose2d_forward_cuda_branchless(
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

  // Weight tensor shape: [in_channels, out_channels, kernel_size, kernel_size]
  int out_channels = weight.size(1);
  int kernel_size = weight.size(2);  // assume square kernel

  // Calculate output dimensions
  int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

  int total_threads = batch_size * out_channels * out_height * out_width;
  int threads = 256;
  int blocks = (total_threads + threads - 1) / threads;

  conv_transpose2d_forward_kernel_branchless<<<blocks, threads>>>(
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
    printf("Error in conv_transpose2d_forward_kernel_branchless: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Wrapper function to handle the possibility of a None bias tensor
torch::Tensor conv_transpose2d_forward_wrapper_branchless(
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
  
  return conv_transpose2d_forward_cuda_branchless(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper_branchless,
        "ConvTranspose2d forward (CUDA) with branchless uniform control flow",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}
