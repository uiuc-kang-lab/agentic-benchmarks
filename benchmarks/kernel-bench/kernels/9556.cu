#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

#define WARP_SIZE 32
#define MAX_KERNEL_SIZE 16

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void conv_transpose2d_forward_kernel_warp_shuffle(
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

    // Calculate output position
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    int output_idx = (blockIdx.x * warps_per_block + warp_id);
    
    if (output_idx >= batch_size * out_channels * out_height * out_width)
        return;

    // Decode output index
    int w_out = output_idx % out_width;
    int temp = output_idx / out_width;
    int h_out = temp % out_height;
    temp /= out_height;
    int o = temp % out_channels;
    int b = temp / out_channels;

    // Precompute base positions
    int base_h = h_out + padding;
    int base_w = w_out + padding;

    // Initialize accumulator
    float partial_sum = (lane_id == 0) ? __ldg(&bias[o]) : 0.0f;

    // Divide work among warp lanes
    for (int c = 0; c < in_channels; c++) {
        for (int p = lane_id; p < kernel_size; p += WARP_SIZE) {
            int h_unscaled = base_h - p * dilation;
            if (h_unscaled % stride == 0) {
                int h_in = h_unscaled / stride;
                if (h_in >= 0 && h_in < in_height) {
                    for (int q = 0; q < kernel_size; q++) {
                        int w_unscaled = base_w - q * dilation;
                        if (w_unscaled % stride == 0) {
                            int w_in = w_unscaled / stride;
                            if (w_in >= 0 && w_in < in_width) {
                                int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
                                int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
                                partial_sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                            }
                        }
                    }
                }
            }
        }
    }

    // Reduce partial sums within warp
    float sum = warpReduceSum(partial_sum);

    // Write result
    if (lane_id == 0) {
        int out_idx = ((b * out_channels + o) * out_height + h_out) * out_width + w_out;
        output[out_idx] = sum;
    }
}

torch::Tensor conv_transpose2d_forward_cuda_warp_shuffle(
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
    int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    int total_elements = batch_size * out_channels * out_height * out_width;
    int threads_per_block = 256; // Multiple of WARP_SIZE
    int warps_per_block = threads_per_block / WARP_SIZE;
    int num_blocks = (total_elements + warps_per_block - 1) / warps_per_block;
    
    conv_transpose2d_forward_kernel_warp_shuffle<<<num_blocks, threads_per_block>>>(
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
    
    return output;
}

torch::Tensor conv_transpose2d_forward_wrapper_warp_shuffle(
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
    
    return conv_transpose2d_forward_cuda_warp_shuffle(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward_wrapper_warp_shuffle,
          "ConvTranspose2d forward with warp shuffle operations (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
}