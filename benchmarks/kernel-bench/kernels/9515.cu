#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Define tile dimensions for output spatial tiling
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// CUDA kernel for 2D transposed convolution using shared memory for weight caching
// This kernel assigns each block a tile of output pixels for a fixed (batch, out_channel) pair.
// It loads the corresponding weight filter from global memory into shared memory exactly once per block,
// then does computation without extra synchronizations (after one __syncthreads() call for consistency).

// Grid dimensions:
//   gridDim.x: number of tiles in the output width direction
//   gridDim.y: number of tiles in the output height direction
//   gridDim.z: batch_size * out_channels
// Block dimensions: dim3(TILE_WIDTH, TILE_HEIGHT)
// Shared memory size: in_channels * kernel_size * kernel_size * sizeof(float)

__global__ void conv_transpose2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation) {

    // Calculate tile indices for output spatial dimensions
    int tile_col = blockIdx.x;  // tile index along width
    int tile_row = blockIdx.y;  // tile index along height
    int bo_index = blockIdx.z;  // combined index for batch and out_channel
    int b = bo_index / out_channels;
    int o = bo_index % out_channels;

    // Compute the global output coordinates for this thread
    int out_x = tile_col * TILE_WIDTH + threadIdx.x;
    int out_y = tile_row * TILE_HEIGHT + threadIdx.y;

    // Declare shared memory for caching the weight filter for this (fixed) output channel 'o'
    // Shared memory size = in_channels * kernel_size * kernel_size floats
    extern __shared__ float sweight[];
    int weight_size = in_channels * kernel_size * kernel_size;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;
    
    // Each thread loads some elements of the weight filter into shared memory
    // Global weight layout: [in_channels, out_channels, kernel_size, kernel_size]
    // For fixed out_channel 'o', index = ((c * out_channels + o) * kernel_size + p)*kernel_size + q
    for (int i = tid; i < weight_size; i += block_size) {
        int c = i / (kernel_size * kernel_size);
        int rem = i % (kernel_size * kernel_size);
        int p = rem / kernel_size;
        int q = rem % kernel_size;
        int global_weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
        sweight[i] = weight[global_weight_idx];
    }
    // Synchronize to ensure shared memory is fully loaded
    __syncthreads();

    // Only compute if the output coordinate is within bounds
    if (out_y < out_height && out_x < out_width) {
        float out_val = bias[o];
        
        // Loop over input channels and kernel spatial dimensions
        for (int c = 0; c < in_channels; c++) {
            for (int p = 0; p < kernel_size; p++) {
                int h_unscaled = out_y + padding - p * dilation;
                if (h_unscaled % stride != 0) continue;
                int h_in = h_unscaled / stride;
                if (h_in < 0 || h_in >= in_height) continue;
                for (int q = 0; q < kernel_size; q++) {
                    int w_unscaled = out_x + padding - q * dilation;
                    if (w_unscaled % stride != 0) continue;
                    int w_in = w_unscaled / stride;
                    if (w_in < 0 || w_in >= in_width) continue;
                    int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
                    int weight_idx = c * (kernel_size * kernel_size) + p * kernel_size + q;
                    out_val += input[input_idx] * sweight[weight_idx];
                }
            }
        }
        int output_idx = ((b * out_channels + o) * out_height + out_y) * out_width + out_x;
        output[output_idx] = out_val;
    }
}

// CUDA launcher function
torch::Tensor conv_transpose2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {

    // Get input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);

    // Weight tensor: [in_channels, out_channels, kernel_size, kernel_size]
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);  // assume square kernel

    // Calculate output dimensions for transposed convolution
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    // Define block and grid dimensions
    dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridDim((out_width + TILE_WIDTH - 1) / TILE_WIDTH,
                 (out_height + TILE_HEIGHT - 1) / TILE_HEIGHT,
                 batch_size * out_channels);

    // Shared memory size: in_channels * kernel_size * kernel_size * sizeof(float)
    size_t sharedMemSize = in_channels * kernel_size * kernel_size * sizeof(float);

    conv_transpose2d_forward_kernel<<<gridDim, blockDim, sharedMemSize>>>(
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

// Wrapper function to handle optional bias (if bias is None, create a zero bias tensor)
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
        "ConvTranspose2d forward (CUDA) with shared memory tiling",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}
