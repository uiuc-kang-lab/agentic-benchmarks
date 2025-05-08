#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Maximum allowed elements in constant memory for weights (64KB / 4 bytes per float)
#define MAX_CONST_WEIGHT_ELEMENTS 16384

// Declare constant memory for weights
__constant__ float d_const_weight[MAX_CONST_WEIGHT_ELEMENTS];

// Macro to define block size: can be tuned to 32, 64, 128, 256, or 512
#ifndef BLOCK_SIZE
  #define BLOCK_SIZE 256
#endif

// This kernel uses a 1D grid where each thread computes one output element.
// The output tensor is in NCHW layout and flattened index is mapped to (n, oc, oh, ow).
__global__ void conv2d_kernel_blocksize_tune(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias,
    int batch, int in_channels, int out_channels,
    int in_height, int in_width, int kernel_size,
    int stride, int padding, int dilation,
    int out_height, int out_width,
    int total_outputs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) return;

    // Map the linear index 'idx' to (n, oc, oh, ow) for NCHW layout
    int ow = idx % out_width;
    int tmp = idx / out_width;
    int oh = tmp % out_height;
    tmp = tmp / out_height;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;

    float sum = 0.0f;
    
    // Loop over all input channels and kernel spatial dimensions
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            int ih = oh * stride - padding + kh * dilation;
            if (ih < 0 || ih >= in_height) continue;
            for (int kw = 0; kw < kernel_size; kw++) {
                int iw = ow * stride - padding + kw * dilation;
                if (iw < 0 || iw >= in_width) continue;

                int input_idx = ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                sum += input[input_idx] * d_const_weight[weight_idx];
            }
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[oc];
    }

    output[idx] = sum;
}


torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    TORCH_CHECK(groups == 1, "Only groups=1 is supported");

    // Input dimensions (N, C, H, W)
    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);

    // Weight dimensions (out_channels, in_channels, kernel_size, kernel_size)
    int out_channels = weight.size(0);
    TORCH_CHECK(weight.size(1) == in_channels, "Mismatch in in_channels between input and weight");
    int kernel_size = weight.size(2);
    TORCH_CHECK(weight.size(3) == kernel_size, "Weight kernel must be square");

    int weight_numel = weight.numel();
    TORCH_CHECK(weight_numel <= MAX_CONST_WEIGHT_ELEMENTS, "Weight tensor has ", weight_numel,
                " elements, which exceeds the constant memory limit of ", MAX_CONST_WEIGHT_ELEMENTS);

    // Compute output dimensions
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width  = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Total number of output elements in NCHW layout
    int total_outputs = batch * out_channels * out_height * out_width;
    auto output = torch::empty({batch, out_channels, out_height, out_width}, input.options());

    // Copy the weight tensor to constant memory
    cudaMemcpyToSymbol(d_const_weight, weight.data_ptr<float>(), weight_numel * sizeof(float));

    // Determine grid size based on BLOCK_SIZE (tunable: 32, 64, 128, 256, or 512)
    int grid_size = (total_outputs + BLOCK_SIZE - 1) / BLOCK_SIZE;

    conv2d_kernel_blocksize_tune<<<grid_size, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        (bias.has_value() ? bias.value().data_ptr<float>() : nullptr),
        batch, in_channels, out_channels,
        in_height, in_width, kernel_size,
        stride, padding, dilation,
        out_height, out_width,
        total_outputs
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with block size tuning");
}
