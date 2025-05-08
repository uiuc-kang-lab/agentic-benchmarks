#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Define warp size and maximum kernel size
#define WARP_SIZE 32
#define MAX_KERNEL_SIZE 16

// This kernel assigns one warp to compute one output element. The warp threads partition the summation over in_channels,
// then use warp-level primitives (__shfl_down_sync) to reduce the partial sums. This eliminates the need for shared memory for
// reduction and speeds up small reductions on NVIDIA H100 GPUs.

__global__ void conv_transpose2d_forward_kernel_warp_reduce(
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

    // Each warp computes one output element.
    unsigned int warpId = (blockIdx.x * (blockDim.x / WARP_SIZE)) + (threadIdx.x / WARP_SIZE);
    int total_outputs = batch_size * out_channels * out_height * out_width;
    if (warpId >= total_outputs) return;

    // Lane id in the warp
    int lane = threadIdx.x % WARP_SIZE;

    // Decode warpId into (b, o, h_out, w_out)
    int tmp = warpId;
    int w_out = tmp % out_width;
    tmp /= out_width;
    int h_out = tmp % out_height;
    tmp /= out_height;
    int o = tmp % out_channels;
    int b = tmp / out_channels;

    int base_h = h_out + padding;
    int base_w = w_out + padding;

    // Precompute valid kernel indices for the h dimension
    int valid_p_count = 0;
    int valid_p[MAX_KERNEL_SIZE];
    int h_in_list[MAX_KERNEL_SIZE];
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

    // Precompute valid kernel indices for the w dimension
    int valid_q_count = 0;
    int valid_q[MAX_KERNEL_SIZE];
    int w_in_list[MAX_KERNEL_SIZE];
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

    // Each warp thread computes a partial sum over a subset of in_channels
    float partial_sum = 0.0f;
    for (int c = lane; c < in_channels; c += WARP_SIZE) {
        for (int i = 0; i < valid_p_count; i++) {
            int p = valid_p[i];
            int h_in = h_in_list[i];
            for (int j = 0; j < valid_q_count; j++) {
                int q = valid_q[j];
                int w_in = w_in_list[j];
                int input_idx = (((b * in_channels + c) * in_height + h_in) * in_width + w_in);
                int weight_idx = (((o * in_channels + c) * kernel_size + p) * kernel_size + q);
                partial_sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
            }
        }
    }

    // Perform warp-level reduction using __shfl_down_sync to sum the partial results
    unsigned int mask = 0xffffffff;
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }

    // Lane 0 adds the bias and writes the final result
    if (lane == 0) {
        float result = partial_sum + __ldg(&bias[o]);
        int output_idx = (((b * out_channels + o) * out_height + h_out) * out_width + w_out);
        output[output_idx] = result;
    }
}

// CUDA forward function using warp-level reduction

torch::Tensor conv_transpose2d_forward_cuda_warp_reduce(
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
    int kernel_size = weight.size(2); // assume square kernel
    
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    // Each warp computes one output element
    int total_outputs = batch_size * out_channels * out_height * out_width;
    int total_threads = total_outputs * WARP_SIZE;
    
    int threads_per_block = 256; // must be a multiple of warp size
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    conv_transpose2d_forward_kernel_warp_reduce<<<blocks, threads_per_block>>>(
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
      printf("Error in conv_transpose2d_forward_kernel_warp_reduce: %s\n", cudaGetErrorString(err));
    }
    
    return output;
}

// Wrapper to handle the possibility of a None bias tensor

torch::Tensor conv_transpose2d_forward_wrapper_warp_reduce(
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
    return conv_transpose2d_forward_cuda_warp_reduce(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward_wrapper_warp_reduce,
          "ConvTranspose2d forward (CUDA) with warp-level primitive reductions",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
}
