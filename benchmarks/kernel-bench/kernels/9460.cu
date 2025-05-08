#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Device function to compute a valid input coordinate for transposed convolution.
__device__ inline bool get_valid_index(int out_coord, int pad, int k, int dilation, int stride, int input_dim, int &in_coord) {
    int unscaled = out_coord + pad - k * dilation;
    if (unscaled % stride != 0)
        return false;
    in_coord = unscaled / stride;
    return (in_coord >= 0 && in_coord < input_dim);
}

// CUDA kernel for 2D transposed convolution using modular device functions.
__global__ void conv_transpose2d_forward_kernel(
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

    // Decode index into (b, o, out_h, out_w)
    int w_out = index % out_width;
    int temp = index / out_width;
    int h_out = temp % out_height;
    temp /= out_height;
    int o = temp % out_channels;
    int b = temp / out_channels;

    float out_val = bias[o];

    // Loop over input channels and kernel elements using the modular get_valid_index function
    for (int c = 0; c < in_channels; ++c) {
        for (int p = 0; p < kernel_size; ++p) {
            int h_in;
            if (!get_valid_index(h_out, padding, p, dilation, stride, in_height, h_in))
                continue;
            for (int q = 0; q < kernel_size; ++q) {
                int w_in;
                if (!get_valid_index(w_out, padding, q, dilation, stride, in_width, w_in))
                    continue;
                int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
                int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
                out_val += input[input_idx] * weight[weight_idx];
            }
        }
    }

    int output_idx = ((b * out_channels + o) * out_height + h_out) * out_width + w_out;
    output[output_idx] = out_val;
}

// CUDA wrapper function to launch the kernel
torch::Tensor conv_transpose2d_forward_cuda(
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

    conv_transpose2d_forward_kernel<<<blocks, threads>>>(
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
        printf("Error in conv_transpose2d_forward_kernel: %s\n", cudaGetErrorString(err));
    }

    return output;
}

// Wrapper to handle the possibility of bias being None
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

    return conv_transpose2d_forward_cuda(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward_wrapper,
          "ConvTranspose2d forward (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
}
