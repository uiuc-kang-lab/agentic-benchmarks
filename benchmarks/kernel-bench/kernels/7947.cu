#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define maximum number of elements allowed in constant memory for weights.
// Adjust MAX_CONST_WEIGHT_ELEMENTS if necessary; here we assume the weight tensor fits in constant memory.
#define MAX_CONST_WEIGHT_ELEMENTS 16384  // 64KB / 4 bytes per float

// Declare constant memory for the weights.
__constant__ float d_const_weight[MAX_CONST_WEIGHT_ELEMENTS];

// CUDA kernel for 2D convolution using constant memory for weight data.
// This kernel assumes the following:
//  - Input tensor in NCHW format
//  - Weight tensor of shape (out_channels, in_channels, kernel_size, kernel_size) with a square kernel
//  - Only supports groups == 1
//  - Supports an optional bias
//  - Dilation is supported

// The kernel launches with a 3D grid where grid.x and grid.y cover the spatial dimensions (output width and height)
// and grid.z covers the combined batch and output channel dimensions.

__global__ void conv2d_kernel_const(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias,
    int batch, int in_channels, int out_channels,
    int in_height, int in_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int out_height,
    int out_width
) {
    // Calculate output spatial coordinates
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;

    // Decode combined index for batch and output channel
    int combo = blockIdx.z; // combo = n * out_channels + oc
    int n = combo / out_channels;
    int oc = combo % out_channels;

    if(n < batch && oc < out_channels && oh < out_height && ow < out_width) {
        float sum = 0.0f;
        // For each input channel
        for (int ic = 0; ic < in_channels; ic++) {
            // Loop over the kernel elements
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int ih = oh * stride - padding + kh * dilation;
                    int iw = ow * stride - padding + kw * dilation;
                    if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                        int input_index = ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                        int weight_index = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                        sum += input[input_index] * d_const_weight[weight_index];
                    }
                }
            }
        }
        if(bias != nullptr) {
            sum += bias[oc];
        }
        int output_index = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
        output[output_index] = sum;
    }
}

// Forward function callable from PyTorch
// This function copies the weight tensor to constant memory and launches the custom CUDA kernel.
// It supports an optional bias and only groups == 1.

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
    if(bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    // This custom kernel only supports groups == 1
    TORCH_CHECK(groups == 1, "conv2d_kernel_const only supports groups == 1");

    // Assume input tensor is in NCHW format
    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);

    // Weight tensor is expected to be of shape: (out_channels, in_channels, kernel_size, kernel_size)
    int out_channels = weight.size(0);
    TORCH_CHECK(weight.size(1) == in_channels, "Mismatch in in_channels between input and weight");
    int kernel_size = weight.size(2);
    TORCH_CHECK(weight.size(3) == kernel_size, "Weight kernel must be square");

    // Check that the weight tensor size fits in constant memory
    int weight_numel = weight.numel();
    TORCH_CHECK(weight_numel <= MAX_CONST_WEIGHT_ELEMENTS,
                "Weight tensor has ", weight_numel,
                " elements, which exceeds the constant memory limit of ", MAX_CONST_WEIGHT_ELEMENTS);

    // Compute output dimensions
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width  = (in_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch, out_channels, out_height, out_width}, input.options());

    // Copy the weight data to constant memory
    cudaMemcpyToSymbol(d_const_weight, weight.data_ptr<float>(), weight_numel * sizeof(float));

    // Setup grid and block dimensions
    const int TILE_X = 16;
    const int TILE_Y = 16;
    dim3 block(TILE_X, TILE_Y, 1);
    dim3 grid(
        (out_width + TILE_X - 1) / TILE_X,
        (out_height + TILE_Y - 1) / TILE_Y,
        batch * out_channels  // Combine batch and output channel into grid.z
    );

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* bias_ptr = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);

    // Launch the CUDA kernel
    conv2d_kernel_const<<<grid, block>>>(
        input_ptr,
        output_ptr,
        bias_ptr,
        batch,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
        dilation,
        out_height,
        out_width
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution using constant memory for weights");
}
