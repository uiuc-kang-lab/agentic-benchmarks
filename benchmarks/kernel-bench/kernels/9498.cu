#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Kernel that computes one output element per warp using warp-level reduction
// Each warp processes the reduction over the input channels and kernel spatial dimensions.
__global__ void conv_transpose2d_forward_kernel_warp_reduction(
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

    // Each warp computes one output element
    // Calculate how many warps per block (assuming blockDim.x is a multiple of 32)
    int warpsPerBlock = blockDim.x >> 5;  // blockDim.x / 32
    int warp_id_in_block = threadIdx.x >> 5; // threadIdx.x / 32
    int lane = threadIdx.x & 31; // threadIdx.x % 32

    // Global warp id
    int global_warp = blockIdx.x * warpsPerBlock + warp_id_in_block;

    // Total number of output elements
    int total_outputs = batch_size * out_channels * out_height * out_width;
    if (global_warp >= total_outputs) return;

    // Decode global_warp into output coordinates: (b, o, out_h, out_w)
    int tmp = global_warp;
    int ow = out_width;
    int oh = out_height;
    int bohw = out_channels * oh * ow;
    int b = tmp / bohw;
    tmp = tmp % bohw;
    int o = tmp / (oh * ow);
    tmp = tmp % (oh * ow);
    int out_h = tmp / ow;
    int out_w = tmp % ow;

    // Each output element is computed as:
    // output[b, o, out_h, out_w] = bias[o] + \sum_{c,p,q} input[b, c, h_in, w_in] * weight[c, o, p, q]
    // where h_in and w_in are derived from out_h, out_w and convolution parameters.

    // The reduction is over total_iter = in_channels * kernel_size * kernel_size
    int total_iter = in_channels * kernel_size * kernel_size;
    float sum = 0.0f;

    // Each lane in the warp processes a subset of the reduction iterations
    for (int i = lane; i < total_iter; i += 32) {
        int c = i / (kernel_size * kernel_size);
        int rem = i % (kernel_size * kernel_size);
        int p = rem / kernel_size;
        int q = rem % kernel_size;

        int h_unscaled = out_h + padding - p * dilation;
        if (h_unscaled % stride != 0) continue;
        int h_in = h_unscaled / stride;
        if (h_in < 0 || h_in >= in_height) continue;

        int w_unscaled = out_w + padding - q * dilation;
        if (w_unscaled % stride != 0) continue;
        int w_in = w_unscaled / stride;
        if (w_in < 0 || w_in >= in_width) continue;

        int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;

        sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
    }

    // Warp-level reduction using __shfl_down_sync
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // The first lane in the warp writes the final result
    if (lane == 0) {
        sum += __ldg(&bias[o]);
        int out_index = ((b * out_channels + o) * out_height + out_h) * out_width + out_w;
        output[out_index] = sum;
    }
}

// CUDA launcher function
torch::Tensor conv_transpose2d_forward_cuda_warp_reduction(
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

    // Calculate output dimensions
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    // Total number of output elements (each computed by one warp)
    int total_outputs = batch_size * out_channels * out_height * out_width;

    // Configure kernel: use a blockDim.x of 128 (i.e., 4 warps per block)
    int block_x = 128;
    int warpsPerBlock = block_x / 32;
    int grid_x = (total_outputs + warpsPerBlock - 1) / warpsPerBlock;

    conv_transpose2d_forward_kernel_warp_reduction<<<grid_x, block_x>>>(
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
        printf("Error in conv_transpose2d_forward_kernel_warp_reduction: %s\n", cudaGetErrorString(err));
    }

    return output;
}

// Wrapper to handle optional bias
torch::Tensor conv_transpose2d_forward_wrapper_warp_reduction(
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

    return conv_transpose2d_forward_cuda_warp_reduction(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward_wrapper_warp_reduction,
          "ConvTranspose2d forward with warp-level reduction (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
}
