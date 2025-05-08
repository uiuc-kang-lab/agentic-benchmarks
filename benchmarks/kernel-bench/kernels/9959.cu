#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define maximum sizes for constant memory buffers
#define MAX_CONSTANT_WEIGHT_SIZE 8192
#define MAX_CONSTANT_BIAS_SIZE 1024

// Block dimensions for the kernel
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8

// Declare constant memory for the weight and bias
__constant__ float c_weight[MAX_CONSTANT_WEIGHT_SIZE];
__constant__ float c_bias[MAX_CONSTANT_BIAS_SIZE];

// CUDA kernel using constant memory for weight and bias
__global__ void depthwise_conv2d_kernel_const(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int channels_per_group,
    bool has_bias
) {
    // Compute output coordinates
    int x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    int b = blockIdx.z / out_channels;
    int c_out = blockIdx.z % out_channels;

    if (x >= out_w || y >= out_h || b >= batch_size) return;

    int g = c_out / channels_per_group;
    int m = c_out % channels_per_group;

    float sum = 0.0f;
    
    // Loop over the kernel spatial dimensions
    for (int kh = 0; kh < kernel_h; kh++) {
        int h_in = y * stride_h - padding_h + kh * dilation_h;
        if (h_in < 0 || h_in >= in_h) continue;
        for (int kw = 0; kw < kernel_w; kw++) {
            int w_in = x * stride_w - padding_w + kw * dilation_w;
            if (w_in < 0 || w_in >= in_w) continue;
            
            int input_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
            int weight_idx = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
            sum += input[input_idx] * c_weight[weight_idx];
        }
    }

    if (has_bias) {
        sum += c_bias[c_out];
    }

    int out_idx = ((b * out_channels + c_out) * out_h + y) * out_w + x;
    output[out_idx] = sum;
}

// Forward function copying weight (and bias) into constant memory before launching the kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = groups * weight.size(1);
    int channels_per_group = out_channels / groups;

    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Copy weight into constant memory; ensure it fits within the allocated constant memory.
    int weight_numel = weight.numel();
    TORCH_CHECK(weight_numel <= MAX_CONSTANT_WEIGHT_SIZE, "Weight tensor too large for constant memory");
    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), weight_numel * sizeof(float));

    bool has_bias = false;
    if (bias.has_value()) {
        int bias_numel = bias->numel();
        TORCH_CHECK(bias_numel <= MAX_CONSTANT_BIAS_SIZE, "Bias tensor too large for constant memory");
        cudaMemcpyToSymbol(c_bias, bias->data_ptr<float>(), bias_numel * sizeof(float));
        has_bias = true;
    }

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 blocks(
        (out_w + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (out_h + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        batch_size * out_channels
    );

    depthwise_conv2d_kernel_const<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        channels_per_group,
        has_bias
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise Conv2D forward using constant memory (CUDA)");
}
