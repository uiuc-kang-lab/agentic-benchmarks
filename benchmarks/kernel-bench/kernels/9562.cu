#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cooperative_groups.h>
#include <pybind11/pybind11.h>

#define MAX_KERNEL_SIZE 16
#define SHARED_WEIGHT_DIM 256  // Tune based on in_channels

namespace cg = cooperative_groups;

__global__ void conv_transpose2d_shared_weights_kernel(
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

  extern __shared__ float shared_weights[];
  cg::thread_block tb = cg::this_thread_block();
  
  const int o = blockIdx.y;
  const int b = blockIdx.z;
  const int h_start = blockIdx.x * blockDim.y;
  const int w_start = threadIdx.y * blockDim.x;

  // Load weights for this output channel into shared memory
  for (int c = threadIdx.x; c < in_channels; c += blockDim.x) {
    for (int p = 0; p < kernel_size; ++p) {
      for (int q = 0; q < kernel_size; ++q) {
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
        shared_weights[(c * kernel_size + p) * kernel_size + q] = __ldg(&weight[weight_idx]);
      }
    }
  }
  tb.sync();

  // Process spatial positions
  for (int h_idx = 0; h_idx < blockDim.y; ++h_idx) {
    int h_out = h_start + h_idx;
    if (h_out >= out_height) continue;
    
    for (int w_idx = 0; w_idx < blockDim.x; ++w_idx) {
      int w_out = w_start + w_idx;
      if (w_out >= out_width) continue;

      float out_val = __ldg(&bias[o]);
      int base_h = h_out + padding;
      int base_w = w_out + padding;

      // Precompute valid positions
      int valid_p[MAX_KERNEL_SIZE], h_in_list[MAX_KERNEL_SIZE], valid_p_count = 0;
      for (int p = 0; p < kernel_size; ++p) {
        int h_unscaled = base_h - p * dilation;
        if (h_unscaled >= 0 && h_unscaled % stride == 0) {
          int h_in = h_unscaled / stride;
          if (h_in < in_height) {
            valid_p[valid_p_count] = p;
            h_in_list[valid_p_count] = h_in;
            valid_p_count++;
          }
        }
      }

      int valid_q[MAX_KERNEL_SIZE], w_in_list[MAX_KERNEL_SIZE], valid_q_count = 0;
      for (int q = 0; q < kernel_size; ++q) {
        int w_unscaled = base_w - q * dilation;
        if (w_unscaled >= 0 && w_unscaled % stride == 0) {
          int w_in = w_unscaled / stride;
          if (w_in < in_width) {
            valid_q[valid_q_count] = q;
            w_in_list[valid_q_count] = w_in;
            valid_q_count++;
          }
        }
      }

      // Accumulate using shared weights
      for (int c = 0; c < in_channels; ++c) {
        for (int i = 0; i < valid_p_count; ++i) {
          int p = valid_p[i];
          int h_in = h_in_list[i];
          for (int j = 0; j < valid_q_count; ++j) {
            int q = valid_q[j];
            int w_in = w_in_list[j];

            float input_val = __ldg(&input[((b * in_channels + c) * in_height + h_in) * in_width + w_in]);
            float weight_val = shared_weights[(c * kernel_size + p) * kernel_size + q];
            out_val += input_val * weight_val;
          }
        }
      }

      // Write output
      output[((b * out_channels + o) * out_height + h_out) * out_width + w_out] = out_val;
    }
  }
}

torch::Tensor conv_transpose2d_forward_shared(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation) {

  // If bias is not provided, initialize it to zeros of shape [out_channels]
  if (!bias.has_value()) {
    int out_channels = weight.size(1);
    bias = torch::zeros({out_channels}, weight.options());
  }

  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int in_height = input.size(2);
  int in_width = input.size(3);
  int out_channels = weight.size(1);
  int kernel_size = weight.size(2);

  int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

  dim3 blocks((out_height + 7) / 8, out_channels, batch_size);
  dim3 threads(32, 8);  // 256 threads per block
  size_t shared_mem = in_channels * kernel_size * kernel_size * sizeof(float);

  conv_transpose2d_shared_weights_kernel<<<blocks, threads, shared_mem>>>(
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
    printf("Error: %s\n", cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_shared,
        "ConvTranspose2D with shared weight optimization");
}
