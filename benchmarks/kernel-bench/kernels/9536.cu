#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Shared memory size for weight caching
#define BLOCK_SIZE 256
#define WARP_SIZE 32

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
    
    __shared__ float shared_weight[WARP_SIZE];
    
    // Calculate output position using coalesced indexing
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_width = out_width * out_channels;
    const int w_idx = tid % total_width;
    const int h_idx = (tid / total_width) % out_height;
    const int b_idx = tid / (total_width * out_height);
    
    if (b_idx >= batch_size) return;
    
    const int o = w_idx / out_width;
    const int w_out = w_idx % out_width;
    const int h_out = h_idx;
    
    float result = 0.0f;
    if (o < out_channels) {
        result = bias[o];
    }
    
    // Process input channels
    for (int c = 0; c < in_channels; c++) {
        for (int p = 0; p < kernel_size; p++) {
            for (int q = 0; q < kernel_size; q++) {
                // Load weight value directly
                float weight_val = 0.0f;
                if (o < out_channels) {
                    weight_val = weight[((c * out_channels + o) * kernel_size + p) * kernel_size + q];
                }
                
                // Calculate input position for transposed convolution
                const int h_unscaled = h_out + padding - p * dilation;
                const int w_unscaled = w_out + padding - q * dilation;
                
                if (h_unscaled % stride == 0 && w_unscaled % stride == 0) {
                    const int h_in = h_unscaled / stride;
                    const int w_in = w_unscaled / stride;
                    
                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        const float in_val = input[((b_idx * in_channels + c) * in_height + h_in) * in_width + w_in];
                        result += in_val * weight_val;
                    }
                }
            }
            }
        }
    }
    
    // Write output using coalesced access pattern
    if (b_idx < batch_size) {
        const int out_idx = ((b_idx * out_channels + o) * out_height + h_out) * out_width + w_out;
        output[out_idx] = result;
    }
}

torch::Tensor conv_transpose2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);
    
    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    const int total_elements = batch_size * out_channels * out_height * out_width;
    const dim3 block_size(BLOCK_SIZE);
    const dim3 grid_size((total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    conv_transpose2d_forward_kernel<<<grid_size, block_size>>>(
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
          "ConvTranspose2d forward (CUDA) with coalesced memory access",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
}