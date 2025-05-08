#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

#define TILE_SIZE 16
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void conv_transpose2d_forward_kernel_tiled(
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

    // Calculate tile indices
    int tile_x = blockIdx.x * TILE_SIZE;
    int tile_y = blockIdx.y * TILE_SIZE;
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    
    // Calculate batch and channel indices
    int b = blockIdx.z / out_channels;
    int o = blockIdx.z % out_channels;

    // Load bias value once per thread
    float bias_val = __ldg(&bias[o]);

    // Process multiple output elements per thread in a tile
    #pragma unroll
    for (int ty = 0; ty < TILE_SIZE; ty += BLOCK_SIZE_Y) {
        int h_out = tile_y + local_y + ty;
        if (h_out >= out_height) continue;

        #pragma unroll
        for (int tx = 0; tx < TILE_SIZE; tx += BLOCK_SIZE_X) {
            int w_out = tile_x + local_x + tx;
            if (w_out >= out_width) continue;

            float out_val = bias_val;

            // Compute input contribution for this output position
            #pragma unroll
            for (int c = 0; c < in_channels; ++c) {
                for (int p = 0; p < kernel_size; ++p) {
                    int h_unscaled = h_out + padding - p * dilation;
                    if (h_unscaled % stride != 0) continue;
                    
                    int h_in = h_unscaled / stride;
                    if (h_in < 0 || h_in >= in_height) continue;

                    for (int q = 0; q < kernel_size; ++q) {
                        int w_unscaled = w_out + padding - q * dilation;
                        if (w_unscaled % stride != 0) continue;
                        
                        int w_in = w_unscaled / stride;
                        if (w_in < 0 || w_in >= in_width) continue;

                        int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
                        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
                        
                        out_val += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                    }
                }
            }

            // Write result
            int output_idx = ((b * out_channels + o) * out_height + h_out) * out_width + w_out;
            output[output_idx] = out_val;
        }
    }
}

torch::Tensor conv_transpose2d_forward_cuda_tiled(
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

    // Calculate grid dimensions for tiled execution
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );

    conv_transpose2d_forward_kernel_tiled<<<grid, threads>>>(
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

torch::Tensor conv_transpose2d_forward_wrapper_tiled(
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
    
    return conv_transpose2d_forward_cuda_tiled(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward_wrapper_tiled,
          "ConvTranspose2d forward (CUDA) with tiled execution",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
}