#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Kernel that maps threads to consecutive output elements to ensure coalesced memory accesses
__global__ void conv_transpose2d_forward_kernel_coalesced_aligned(
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

    // Compute global output coordinates; ensure that threads in a warp produce consecutive output locations
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;  // fastest varying across warps
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int b_o = blockIdx.z; // combined index for batch and output channel

    if (out_w >= out_width || out_h >= out_height) return;

    int o = b_o % out_channels;
    int b = b_o / out_channels;

    // Read bias using aligned load
    float sum = __ldg(&bias[o]);

    // Loop over input channels and kernel spatial dimensions
    #pragma unroll
    for (int c = 0; c < in_channels; ++c) {
        #pragma unroll
        for (int p = 0; p < kernel_size; ++p) {
            int h_unscaled = out_h + padding - p * dilation;
            if (h_unscaled % stride != 0) continue;
            int h_in = h_unscaled / stride;
            if (h_in < 0 || h_in >= in_height) continue;
            
            #pragma unroll
            for (int q = 0; q < kernel_size; ++q) {
                int w_unscaled = out_w + padding - q * dilation;
                if (w_unscaled % stride != 0) continue;
                int w_in = w_unscaled / stride;
                if (w_in < 0 || w_in >= in_width) continue;
                
                int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
                int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
                
                // Use __ldg to load read-only data in an aligned manner
                sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
            }
        }
    }
    
    int output_idx = ((b * out_channels + o) * out_height + out_h) * out_width + out_w;
    output[output_idx] = sum;
}

// CUDA launcher function
torch::Tensor conv_transpose2d_forward_cuda_coalesced_aligned(
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
    int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    // Configure thread block to map consecutive threads to adjacent output elements (ensuring coalesced writes).
    dim3 block(32, 8);
    dim3 grid((out_width + block.x - 1) / block.x,
              (out_height + block.y - 1) / block.y,
              batch_size * out_channels);

    conv_transpose2d_forward_kernel_coalesced_aligned<<<grid, block>>>(
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
        printf("Error in conv_transpose2d_forward_kernel_coalesced_aligned: %s\n", cudaGetErrorString(err));
    }
    
    return output;
}

// Wrapper function to handle optional bias (None case)
torch::Tensor conv_transpose2d_forward_wrapper_coalesced_aligned(
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

    return conv_transpose2d_forward_cuda_coalesced_aligned(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward_wrapper_coalesced_aligned,
          "ConvTranspose2d forward with coalesced and aligned memory accesses (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
}
