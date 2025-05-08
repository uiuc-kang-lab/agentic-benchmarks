#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// These macros allow tuning the block dimensions at compile time.
#ifndef BLOCK_DIM_X
#define BLOCK_DIM_X 16
#endif

#ifndef BLOCK_DIM_Y
#define BLOCK_DIM_Y 16
#endif

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel performs a 2D convolution with tunable block sizes.
// The block dimensions (BLOCK_DIM_X x BLOCK_DIM_Y) are set via compile-time macros to allow experimentation with different configurations
// such as 32, 64, 128, 256, or 512 threads per block. By adjusting these parameters, optimal occupancy and memory
// coalescing for the NVIDIA H100 GPU can be achieved while maintaining correct results.
__global__ void conv2d_tuned_blocksize_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // May be nullptr if bias is not provided
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_height,
    int kernel_width,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Determine the output spatial coordinates
    int w = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    int h = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
    int oc = blockIdx.z;  // Each block in the z-dimension corresponds to an output channel

    if (w < out_width && h < out_height && oc < out_channels) {
        int group_out_channels = out_channels / groups;
        int group = oc / group_out_channels;
        int in_channels_per_group = in_channels / groups;

        // Iterate over the batch dimension
        for (int b = 0; b < batch_size; ++b) {
            float sum = 0.0f;
            for (int c = 0; c < in_channels_per_group; ++c) {
                int input_channel = group * in_channels_per_group + c;
                for (int kh = 0; kh < kernel_height; ++kh) {
                    for (int kw = 0; kw < kernel_width; ++kw) {
                        int in_y = h * stride - padding + kh * dilation;
                        int in_x = w * stride - padding + kw * dilation;
                        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                            int input_idx = ((b * in_channels + input_channel) * in_height + in_y) * in_width + in_x;
                            int weight_idx = (((oc * in_channels_per_group + c) * kernel_height) + kh) * kernel_width + kw;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            if (bias != nullptr) {
                sum += bias[oc];
            }
            int output_idx = ((b * out_channels + oc) * out_height + h) * out_width + w;
            output[output_idx] = sum;
        }
    }
}

// Forward function to launch the convolution kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    int out_width  = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto options = x.options();
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, options);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    // Setup grid dimensions based on the tunable block sizes
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 grid((out_width  + block.x - 1) / block.x,
              (out_height + block.y - 1) / block.y,
              out_channels);

    conv2d_tuned_blocksize_kernel<<<grid, block>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride,
        padding,
        dilation,
        groups
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution with Tunable Block Sizes");
}
