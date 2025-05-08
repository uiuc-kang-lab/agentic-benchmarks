#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Kernel: Each warp computes one output element using warp-level reduction via __shfl_down_sync
__global__ void conv_transpose2d_forward_kernel_warp(
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

    // Global thread ID and warp-level indices
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = global_thread_id / 32;  // one warp = 32 threads
    int lane = global_thread_id % 32;

    int total_output = batch_size * out_channels * out_height * out_width;
    if (warpId >= total_output) return;

    // Decode warpId into (b, o, out_h, out_w)
    int tmp = warpId;
    int w_out = tmp % out_width;
    tmp /= out_width;
    int h_out = tmp % out_height;
    tmp /= out_height;
    int o = tmp % out_channels;
    int b = tmp / out_channels;

    // Each thread in the warp computes partial sum over a subset of input channels
    float partial_sum = 0.0f;
    for (int c = lane; c < in_channels; c += 32) {
        for (int p = 0; p < kernel_size; p++) {
            int h_unscaled = h_out + padding - p * dilation;
            if (h_unscaled % stride != 0) continue;
            int h_in = h_unscaled / stride;
            if (h_in < 0 || h_in >= in_height) continue;
            for (int q = 0; q < kernel_size; q++) {
                int w_unscaled = w_out + padding - q * dilation;
                if (w_unscaled % stride != 0) continue;
                int w_in = w_unscaled / stride;
                if (w_in < 0 || w_in >= in_width) continue;
                int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
                int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
                partial_sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
            }
        }
    }

    // Warp-level reduction using __shfl_down_sync for summing the partial sums
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    // Lane 0 adds bias and writes the final output
    if (lane == 0) {
        partial_sum += __ldg(&bias[o]);
        int output_idx = ((b * out_channels + o) * out_height + h_out) * out_width + w_out;
        output[output_idx] = partial_sum;
    }
}

// CUDA launcher function
torch::Tensor conv_transpose2d_forward_cuda_warp(
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
    int kernel_size = weight.size(2);  // assume square kernel

    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    // Each warp computes one output element
    int total_output = batch_size * out_channels * out_height * out_width;
    int total_warps = total_output;
    int threadsPerBlock = 128;  // must be a multiple of 32
    int total_threads = total_warps * 32;
    int blocks = (total_threads + threadsPerBlock - 1) / threadsPerBlock;

    conv_transpose2d_forward_kernel_warp<<<blocks, threadsPerBlock>>>(
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
        printf("Error in conv_transpose2d_forward_kernel_warp: %s\n", cudaGetErrorString(err));
    }

    return output;
}

// Wrapper to handle optional bias
torch::Tensor conv_transpose2d_forward_wrapper_warp(
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

    return conv_transpose2d_forward_cuda_warp(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward_wrapper_warp,
          "ConvTranspose2d forward with warp-level reduction (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
}
