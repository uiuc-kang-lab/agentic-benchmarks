#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Device function to decode a linear index into (w_out, h_out, o, b)
__device__ inline void decode_index(int index, int out_width, int out_height, int out_channels,
                                    int &w_out, int &h_out, int &o, int &b) {
    w_out = index % out_width;
    int temp = index / out_width;
    h_out = temp % out_height;
    temp /= out_height;
    o = temp % out_channels;
    b = temp / out_channels;
}

// Device function to compute the contribution from one input channel
__device__ inline float compute_channel_contribution(
    const float* input,
    const float* shared_weight,
    int b,
    int c,
    int o,
    int in_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int out_h,
    int out_w,
    int pad,
    int dilation,
    int stride) {

    float sum = 0.0f;
    for (int p = 0; p < kernel_size; ++p) {
        int h_unscaled = out_h + pad - p * dilation;
        if (h_unscaled % stride != 0)
            continue;
        int h_in = h_unscaled / stride;
        if (h_in < 0 || h_in >= in_height)
            continue;
        for (int q = 0; q < kernel_size; ++q) {
            int w_unscaled = out_w + pad - q * dilation;
            if (w_unscaled % stride != 0)
                continue;
            int w_in = w_unscaled / stride;
            if (w_in < 0 || w_in >= in_width)
                continue;
            int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
            int weight_shared_idx = c * (kernel_size * kernel_size) + p * kernel_size + q;
            sum += input[input_idx] * shared_weight[weight_shared_idx];
        }
    }
    return sum;
}

// Optimized CUDA kernel using shared memory and modular functions
__global__ void conv_transpose2d_forward_kernel_optimized(
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

    extern __shared__ float shared_weight[];
    int weight_count = in_channels * kernel_size * kernel_size;
    int tid = threadIdx.x;
    for (int i = tid; i < weight_count; i += blockDim.x) {
        int tmp = i;
        int q = tmp % kernel_size;
        tmp /= kernel_size;
        int p = tmp % kernel_size;
        int c = tmp / kernel_size;
        int weight_idx = ((c * out_channels + blockIdx.y) * kernel_size + p) * kernel_size + q;
        shared_weight[i] = weight[weight_idx];
    }
    __syncthreads();

    int w_out, h_out, o, b;
    decode_index(index, out_width, out_height, out_channels, w_out, h_out, o, b);

    float out_val = bias[o];
    for (int c = 0; c < in_channels; ++c) {
        out_val += compute_channel_contribution(input, shared_weight, b, c, o,
                                                  in_channels, in_height, in_width,
                                                  kernel_size, h_out, w_out,
                                                  padding, dilation, stride);
    }
    int output_idx = ((b * out_channels + o) * out_height + h_out) * out_width + w_out;
    output[output_idx] = out_val;
}

// CUDA launcher function
torch::Tensor conv_transpose2d_forward_cuda_optimized(
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

    int total_threads = batch_size * out_channels * out_height * out_width;
    int threads = 1024;
    int blocks = (total_threads + threads - 1) / threads;

    int weight_count = in_channels * kernel_size * kernel_size;
    size_t shared_mem_size = weight_count * sizeof(float);

    conv_transpose2d_forward_kernel_optimized<<<blocks, threads, shared_mem_size>>>(
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
        printf("Error in conv_transpose2d_forward_kernel_optimized: %s\n", cudaGetErrorString(err));
    }

    return output;
}

// Wrapper to handle the case when bias is None
torch::Tensor conv_transpose2d_forward_wrapper_optimized(
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

    return conv_transpose2d_forward_cuda_optimized(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward_wrapper_optimized,
          "ConvTranspose2d forward optimized (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
}