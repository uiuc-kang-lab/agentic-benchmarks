#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Constant memory for weights and bias (limit sizes for H100)
__constant__ float d_weight[12288];
__constant__ float d_bias[1024];

__global__ void conv2d_uniform_kernel(
    const float * __restrict__ input,
    float * __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int out_height,
    int out_width) {
    
    int bc = blockIdx.z;  // Combined batch & output channel
    int b = bc / out_channels;
    int channel = bc % out_channels;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= batch_size || row >= out_height || col >= out_width || channel >= out_channels) return;

    float sum = 0.0f;
    if (bias.has_value()) {
        sum = d_bias[channel];
    }
    const int in_row_origin = row * stride - padding;
    const int in_col_origin = col * stride - padding;

    // Compute kernel bounds without conditional branches
    const int kh_start = max(0, (-in_row_origin + dilation - 1) / dilation);
    const int kh_end = min(kernel_size, (in_height - in_row_origin + dilation - 1) / dilation);
    const int kw_start = max(0, (-in_col_origin + dilation - 1) / dilation);
    const int kw_end = min(kernel_size, (in_width - in_col_origin + dilation - 1) / dilation);

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = kh_start; kh < kh_end; ++kh) {
            const int in_row = in_row_origin + kh * dilation;
            for (int kw = kw_start; kw < kw_end; ++kw) {
                const int in_col = in_col_origin + kw * dilation;
                const int input_idx = ((b * in_channels + ic) * in_height + in_row) * in_width + in_col;
                const int weight_idx = ((channel * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                sum += __ldg(&input[input_idx]) * d_weight[weight_idx];
            }
        }
    }

    const int output_idx = ((b * out_channels + channel) * out_height + row) * out_width + col;
    output[output_idx] = sum;
}

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
    if (bias.has_value()) CHECK_INPUT(bias.value());

    // Dimension calculations
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    // Calculate output dimensions
    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Allocate output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    // Copy parameters to constant memory
    cudaMemcpyToSymbol(d_weight, weight.data_ptr<float>(), weight.numel() * sizeof(float));
    if (bias.has_value()) {
        cudaMemcpyToSymbol(d_bias, bias.value().data_ptr<float>(), bias.value().numel() * sizeof(float));
    }

    // Optimized block dimensions for H100
    dim3 block(32, 8);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        batch_size * out_channels
    );

    conv2d_uniform_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        out_height,
        out_width
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel execution failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Uniform 2D convolution with minimized warp divergence");
}