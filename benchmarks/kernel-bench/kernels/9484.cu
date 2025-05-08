#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// This kernel assigns one warp (32 threads) per output element. Each thread in the warp computes a partial sum over a slice of input channels,
// and then a warp-level reduction using __shfl_down_sync is performed to sum these partial results. Finally, bias is added and the result is written to global memory.

__global__ void conv_transpose2d_forward_kernel_warplevel(
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

    // Each warp (32 threads) computes one output element
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    // Total number of output elements
    int total_output = batch_size * out_channels * out_height * out_width;
    if (warp_id >= total_output) return;

    // Decode the output index from warp_id
    int temp = warp_id;
    int out_w = temp % out_width;
    temp /= out_width;
    int out_h = temp % out_height;
    temp /= out_height;
    int o = temp % out_channels;
    int b = temp / out_channels;

    // Each thread in the warp computes a partial sum over a portion of the input channels
    float partial_sum = 0.0f;
    for (int c = lane; c < in_channels; c += 32) {
        // Loop over kernel's height dimension
        for (int p = 0; p < kernel_size; p++) {
            int h_unscaled = out_h + padding - p * dilation;
            if (h_unscaled % stride != 0) continue;
            int h_in = h_unscaled / stride;
            if (h_in < 0 || h_in >= in_height) continue;
            // Loop over kernel's width dimension
            for (int q = 0; q < kernel_size; q++) {
                int w_unscaled = out_w + padding - q * dilation;
                if (w_unscaled % stride != 0) continue;
                int w_in = w_unscaled / stride;
                if (w_in < 0 || w_in >= in_width) continue;
                
                int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
                int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
                partial_sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    // Perform warp-level reduction using __shfl_down_sync
    // Full mask for active threads in the warp
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    // The first lane of the warp writes the result
    if (lane == 0) {
        float result = bias[o] + partial_sum;
        int output_idx = ((b * out_channels + o) * out_height + out_h) * out_width + out_w;
        output[output_idx] = result;
    }
}


// CUDA launcher function
torch::Tensor conv_transpose2d_forward_cuda_warplevel(
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
    int kernel_size = weight.size(2);  // assuming square kernel

    // Calculate output dimensions
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    // Total number of output elements (each computed by one warp)
    int total_output = batch_size * out_channels * out_height * out_width;
    int total_threads = total_output * 32;  // 32 threads per warp

    // Launch configuration: choose a block size that's a multiple of 32, e.g., 256 threads per block
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    conv_transpose2d_forward_kernel_warplevel<<<grid_size, block_size>>>(
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
        printf("Error in conv_transpose2d_forward_kernel_warplevel: %s\n", cudaGetErrorString(err));
    }

    return output;
}

// Wrapper function to handle the possibility that the bias is None
torch::Tensor conv_transpose2d_forward_wrapper_warplevel(
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

    return conv_transpose2d_forward_cuda_warplevel(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward_wrapper_warplevel,
          "ConvTranspose2d forward using warp-level primitives (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
}
