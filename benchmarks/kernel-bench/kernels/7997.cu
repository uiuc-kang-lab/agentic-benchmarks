#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void warp_reduce_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias, bool use_bias,
    float* __restrict__ output,
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

    const int idx = blockIdx.x;
    if (idx >= batch_size * out_channels * out_height * out_width) return;

    // Unpack output coordinates
    int b = idx / (out_channels * out_height * out_width);
    int rem = idx % (out_channels * out_height * out_width);
    int ch = rem / (out_height * out_width);
    rem = rem % (out_height * out_width);
    int row = rem / out_width;
    int col = rem % out_width;

    float sum = 0.0f;
    const int in_row_origin = row * stride - padding;
    const int in_col_origin = col * stride - padding;

    // Distribute input channels across warp threads
    for (int ic = threadIdx.x; ic < in_channels; ic += blockDim.x) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            const int in_row = in_row_origin + kh * dilation;
            if (in_row < 0 || in_row >= in_height) continue;
            
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int in_col = in_col_origin + kw * dilation;
                if (in_col < 0 || in_col >= in_width) continue;

                const int input_idx = ((b * in_channels + ic) * in_height + in_row) * in_width + in_col;
                const int weight_idx = ((ch * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
            }
        }
    }

    // Warp-level tree reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (threadIdx.x == 0) {
        output[((b * out_channels + ch) * out_height + row) * out_width + col] = 
            sum + (bias ? __ldg(&bias[ch]) : 0.0f);
    }
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

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    const dim3 block(32);  // Full warp per output
    const size_t total_outputs = batch_size * out_channels * out_height * out_width;
    const dim3 grid((total_outputs + block.x - 1) / block.x);

    warp_reduce_conv_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
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

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Conv2D with warp-level channel reduction");
}