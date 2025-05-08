#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

#define MAX_KERNEL_SIZE 16
#define MAX_CONSTANT_WEIGHTS 16384  // 64KB / sizeof(float)

// Constant memory for weights - 64KB on H100
__constant__ float const_weight[MAX_CONSTANT_WEIGHTS];

__global__ void conv_transpose2d_forward_kernel_constant(
    const float* __restrict__ input,
    const float* __restrict__ weight,  // Fallback for large weights
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
    int dilation,
    bool use_const_mem) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * out_channels * out_height * out_width;
  if (index >= total)
    return;

  // Decode index
  int w_out = index % out_width;
  int temp = index / out_width;
  int h_out = temp % out_height;
  temp /= out_height;
  int o = temp % out_channels;
  int b = temp / out_channels;

  // Precompute base indices
  int base_h = h_out + padding;
  int base_w = w_out + padding;

  // Precompute valid kernel indices for h dimension
  int valid_p_count = 0;
  int valid_p[MAX_KERNEL_SIZE];
  int h_in_list[MAX_KERNEL_SIZE];
  
  #pragma unroll
  for (int p = 0; p < kernel_size; p++) {
    int p_dilated = p * dilation;
    if (base_h >= p_dilated && ((base_h - p_dilated) % stride) == 0) {
      int h_in = (base_h - p_dilated) / stride;
      if (h_in < in_height) {
        valid_p[valid_p_count] = p;
        h_in_list[valid_p_count] = h_in;
        valid_p_count++;
      }
    }
  }

  // Precompute valid kernel indices for w dimension
  int valid_q_count = 0;
  int valid_q[MAX_KERNEL_SIZE];
  int w_in_list[MAX_KERNEL_SIZE];
  
  #pragma unroll
  for (int q = 0; q < kernel_size; q++) {
    int q_dilated = q * dilation;
    if (base_w >= q_dilated && ((base_w - q_dilated) % stride) == 0) {
      int w_in = (base_w - q_dilated) / stride;
      if (w_in < in_width) {
        valid_q[valid_q_count] = q;
        w_in_list[valid_q_count] = w_in;
        valid_q_count++;
      }
    }
  }

  float out_val = __ldg(&bias[o]);

  for (int c = 0; c < in_channels; ++c) {
    #pragma unroll
    for (int i = 0; i < valid_p_count; i++) {
      int p = valid_p[i];
      int h_in = h_in_list[i];
      
      #pragma unroll
      for (int j = 0; j < valid_q_count; j++) {
        int q = valid_q[j];
        int w_in = w_in_list[j];
        
        int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
        
        float weight_val;
        if (use_const_mem) {
          weight_val = const_weight[weight_idx];
        } else {
          weight_val = __ldg(&weight[weight_idx]);
        }
        
        out_val += __ldg(&input[input_idx]) * weight_val;
      }
    }
  }

  output[((b * out_channels + o) * out_height + h_out) * out_width + w_out] = out_val;
}

torch::Tensor conv_transpose2d_forward_cuda_constant(
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

  // Determine if weight tensor fits in constant memory
  int weight_size = weight.numel() * sizeof(float);
  bool use_const_mem = weight_size <= (MAX_CONSTANT_WEIGHTS * sizeof(float));
  
  if (use_const_mem) {
    // Copy weights to constant memory
    cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), weight_size);
  }

  int total_threads = batch_size * out_channels * out_height * out_width;
  int threads = 256;
  int blocks = (total_threads + threads - 1) / threads;

  conv_transpose2d_forward_kernel_constant<<<blocks, threads>>>(
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
      dilation,
      use_const_mem);

  return output;
}

torch::Tensor conv_transpose2d_forward_wrapper_constant(
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
  
  return conv_transpose2d_forward_cuda_constant(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper_constant,
        "ConvTranspose2d forward (CUDA) with constant memory optimization",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}