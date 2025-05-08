#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Maximum kernel size (assumed to be appropriate)
#define MAX_KERNEL_SIZE 16

// Optimized CUDA kernel for 2D transposed convolution with improved thread and block mappings.
// Precomputes valid kernel positions for dimensions to minimize warp divergence.
__global__ void conv_transpose2d_forward_kernel_thread_block_map(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
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

    // Calculate global coordinates for batch index
    int b = blockIdx.z;
    int o = blockIdx.y % out_channels;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;

    float out_val = (w_out < out_width && h_out < out_height) ? __ldg(&bias[o]) : 0.0f;

    if (w_out >= out_width || h_out >= out_height) return;

    for (int c = 0; c < in_channels; ++c) {
        int base_h = h_out + padding;
        int base_w = w_out + padding;

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

        // Iterate using precomputed valid indices
        for (int i = 0; i < valid_p_count; i++) {
            int p = valid_p[i];
            int h_in = h_in_list[i];
            for (int j = 0; j < valid_q_count; j++) {
                int q = valid_q[j];
                int w_in = w_in_list[j];

                int input_idx = (((b * in_channels + c) * in_height) + h_in) * in_width + w_in;
                int weight_idx = (((c * out_channels + o) * kernel_size + p) * kernel_size) + q;

                out_val += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
            }
        }
    }

    int output_idx = (((b * out_channels) + o) * out_height + h_out) * out_width + w_out;
    output[output_idx] = out_val;
}

// CUDA forward function implementation with updated block and thread mapping
torch::Tensor conv_transpose2d_forward_cuda_thread_block_map(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {

    // Get shape information
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);

    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);

    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    const int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    dim3 threads(16, 16);
    dim3 blocks((out_width + 16 - 1) / 16, (out_height + 16 - 1) / 16, batch_size);

    conv_transpose2d_forward_kernel_thread_block_map<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
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
        printf("Error in conv_transpose2d_forward_kernel_thread_block_map: %s\n", cudaGetErrorString(err));
    }

    return output;
}

// Wrapper function with potential bias handling
torch::Tensor conv_transpose2d_forward_wrapper_thread_block_map(
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
    return conv_transpose2d_forward_cuda_thread_block_map(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward_wrapper_thread_block_map,
          "ConvTranspose2d forward (CUDA) with improved thread and block mappings",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
}
