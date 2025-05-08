#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Optimized CUDA kernel for ConvTranspose2d using shared memory for weight tile.
// Each block processes a single output channel for a given batch,
// and preloads the corresponding weight tile into shared memory to reduce global memory accesses.
// The __syncthreads() is used only once after loading the weight tile for consistency.

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

    // Decode batch index and output channel from blockIdx.z
    int batch_channel = blockIdx.z;
    int b = batch_channel / out_channels;
    int o = batch_channel % out_channels;

    // Compute output spatial coordinates
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    if (h_out >= out_height || w_out >= out_width)
        return;

    // Allocate shared memory for the weight tile for this output channel 'o'.
    extern __shared__ float shared_weight[];  // Size: in_channels * kernel_size * kernel_size floats

    int weight_tile_size = in_channels * kernel_size * kernel_size;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y;

    // Each thread loads part of the weight tile from global memory into shared memory.
    for (int i = tid; i < weight_tile_size; i += block_threads) {
        int c = i / (kernel_size * kernel_size);
        int rem = i % (kernel_size * kernel_size);
        int p = rem / kernel_size;
        int q = rem % kernel_size;
        // Global weight index for fixed output channel o.
        shared_weight[i] = weight[((c * out_channels + o) * kernel_size + p) * kernel_size + q];
    }
    __syncthreads();

    float out_val = bias[o];  // Initialize with bias

    // Accumulate contributions from input channels and kernel window.
    for (int c = 0; c < in_channels; ++c) {
        for (int p = 0; p < kernel_size; ++p) {
            int h_unscaled = h_out + padding - p * dilation;
            if (h_unscaled % stride != 0)
                continue;
            int h_in = h_unscaled / stride;
            if (h_in < 0 || h_in >= in_height)
                continue;
            for (int q = 0; q < kernel_size; ++q) {
                int w_unscaled = w_out + padding - q * dilation;
                if (w_unscaled % stride != 0)
                    continue;
                int w_in = w_unscaled / stride;
                if (w_in < 0 || w_in >= in_width)
                    continue;
                int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
                // Use shared memory for the weight: index computed as c * (kernel_size*kernel_size) + p * kernel_size + q
                int weight_idx = ((c * kernel_size) + p) * kernel_size + q;
                out_val += input[input_idx] * shared_weight[weight_idx];
            }
        }
    }

    int output_idx = ((b * out_channels + o) * out_height + h_out) * out_width + w_out;
    output[output_idx] = out_val;
}


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

    // Calculate output dimensions
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_width  = (in_width  - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    // Set up block dimensions (using 2D blocks for spatial dimensions)
    dim3 block(16, 16);
    // Grid dimensions: x covers out_width, y covers out_height, z covers batch * out_channels
    dim3 grid((out_width + block.x - 1) / block.x,
              (out_height + block.y - 1) / block.y,
              batch_size * out_channels);

    // Compute shared memory size in bytes for the weight tile
    size_t shared_mem_size = in_channels * kernel_size * kernel_size * sizeof(float);

    conv_transpose2d_forward_kernel_optimized<<<grid, block, shared_mem_size>>>(
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

// Wrapper to handle the possibility of bias being None
torch::Tensor conv_transpose2d_forward_wrapper(
    torch::Tensor input,
    torch::Tensor weight,
    pybind11::object bias_obj,  // using py::object to accept None
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
          "Optimized ConvTranspose2d forward (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
}
