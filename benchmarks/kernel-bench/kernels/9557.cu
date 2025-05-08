#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Define warp size constant
#define WARP_SIZE 32
#define MAX_KERNEL_SIZE 16

// Declare constant memory for frequently accessed parameters
__constant__ int c_stride;
__constant__ int c_padding;
__constant__ int c_dilation;
__constant__ int c_in_height;
__constant__ int c_in_width;
__constant__ int c_kernel_size;

__global__ void conv_transpose2d_forward_kernel_warp_aligned(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int out_height,
    int out_width) {

    // Calculate global thread index and warp index
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int total_warps = (blockDim.x * gridDim.x) / WARP_SIZE;

    // Calculate total output elements
    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int elements_per_warp = WARP_SIZE;

    // Process elements in warp-aligned groups
    for (int base_idx = warp_id * elements_per_warp; base_idx < total_elements; base_idx += total_warps * elements_per_warp) {
        int idx = base_idx + lane_id;
        if (idx >= total_elements) continue;

        // Decode output position
        const int w_out = idx % out_width;
        int temp = idx / out_width;
        const int h_out = temp % out_height;
        temp /= out_height;
        const int o = temp % out_channels;
        const int b = temp / out_channels;

        float result = __ldg(&bias[o]);

        // Pre-calculate base positions
        const int base_h = h_out + c_padding;
        const int base_w = w_out + c_padding;

        // Process input contributions
        #pragma unroll 4
        for (int c = 0; c < in_channels; ++c) {
            const int input_batch_offset = b * in_channels * c_in_height * c_in_width;
            const int input_channel_offset = c * c_in_height * c_in_width;
            const int weight_offset = (c * out_channels + o) * c_kernel_size * c_kernel_size;

            #pragma unroll
            for (int p = 0; p < c_kernel_size; ++p) {
                const int h_unscaled = base_h - p * c_dilation;
                const int h_in = h_unscaled / c_stride;
                
                // Check h_in validity once for the entire row
                if (h_unscaled >= 0 && (h_unscaled % c_stride) == 0 && h_in < c_in_height) {
                    #pragma unroll
                    for (int q = 0; q < c_kernel_size; ++q) {
                        const int w_unscaled = base_w - q * c_dilation;
                        const int w_in = w_unscaled / c_stride;

                        if (w_unscaled >= 0 && (w_unscaled % c_stride) == 0 && w_in < c_in_width) {
                            const int input_idx = input_batch_offset + input_channel_offset + 
                                                h_in * c_in_width + w_in;
                            const int weight_idx = weight_offset + p * c_kernel_size + q;
                            
                            result += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                        }
                    }
                }
            }
        }

        // Write result
        if (idx < total_elements) {
            output[idx] = result;
        }
    }
}

torch::Tensor conv_transpose2d_forward_cuda_warp_aligned(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {

    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    // Calculate output dimensions
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    // Copy constants to constant memory
    cudaMemcpyToSymbol(c_stride, &stride, sizeof(int));
    cudaMemcpyToSymbol(c_padding, &padding, sizeof(int));
    cudaMemcpyToSymbol(c_dilation, &dilation, sizeof(int));
    cudaMemcpyToSymbol(c_in_height, &in_height, sizeof(int));
    cudaMemcpyToSymbol(c_in_width, &in_width, sizeof(int));
    cudaMemcpyToSymbol(c_kernel_size, &kernel_size, sizeof(int));

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    // Calculate grid and block dimensions
    const int threads_per_block = 256; // Multiple of WARP_SIZE
    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose2d_forward_kernel_warp_aligned<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        out_height,
        out_width);

    return output;
}

torch::Tensor conv_transpose2d_forward_wrapper_warp_aligned(
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

    return conv_transpose2d_forward_cuda_warp_aligned(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward_wrapper_warp_aligned,
          "ConvTranspose2d forward (CUDA) with warp-aligned processing",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
}