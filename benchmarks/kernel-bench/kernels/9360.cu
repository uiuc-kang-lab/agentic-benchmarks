#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHANNELS_PER_BLOCK 4
#define MAX_KERNEL_SIZE 7
#define MAX_IN_CHANNELS 64
#define MAX_OUT_CHANNELS 64

// Constant memory for weights and bias
__constant__ float const_weight[MAX_OUT_CHANNELS * MAX_IN_CHANNELS * MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
__constant__ float const_bias[MAX_OUT_CHANNELS];

__global__ void conv2d_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int height_out,
    int width_out,
    int stride,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    bool has_bias) {

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Block dimensions and output position
    const int blockDimX = blockDim.x;
    const int blockDimY = blockDim.y;
    const int h_out = by * blockDimY + ty;
    const int w_out = bx * blockDimX + tx;

    // Batch and channel group handling
    const int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    const int b = bz / groups_per_batch;
    const int g = bz % groups_per_batch;
    const int oc_start = g * CHANNELS_PER_BLOCK;

    if (h_out >= height_out || w_out >= width_out || b >= batch_size) return;

    // Initialize accumulation registers
    float sums[CHANNELS_PER_BLOCK] = {0.0f};
    
    // Load bias from constant memory if present
    #pragma unroll
    for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
        const int global_oc = oc_start + i;
        if (global_oc < out_channels) {
            sums[i] = has_bias ? const_bias[global_oc] : 0.0f;
        }
    }

    // Main convolution loop
    #pragma unroll 4
    for (int ic = 0; ic < in_channels; ic++) {
        #pragma unroll
        for (int kh = 0; kh < kernel_h; kh++) {
            const int h_in = h_out * stride + kh * dilation_h - pad_h;
            if (h_in >= 0 && h_in < input_height) {
                #pragma unroll
                for (int kw = 0; kw < kernel_w; kw++) {
                    const int w_in = w_out * stride + kw * dilation_w - pad_w;
                    if (w_in >= 0 && w_in < input_width) {
                        // Load input value using read-only cache
                        const float x_val = __ldg(&x[b * in_channels * input_height * input_width +
                                                   ic * input_height * input_width +
                                                   h_in * input_width + w_in]);

                        // Accumulate for each output channel using constant memory weights
                        #pragma unroll
                        for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
                            const int global_oc = oc_start + i;
                            if (global_oc < out_channels) {
                                const int weight_idx = global_oc * in_channels * kernel_h * kernel_w +
                                                     ic * kernel_h * kernel_w +
                                                     kh * kernel_w + kw;
                                sums[i] += x_val * const_weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
        const int global_oc = oc_start + i;
        if (global_oc < out_channels) {
            const int out_idx = b * out_channels * height_out * width_out +
                              global_oc * height_out * width_out +
                              h_out * width_out + w_out;
            output[out_idx] = sums[i];
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    std::tuple<int, int> padding,
    std::tuple<int, int> dilation) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    TORCH_CHECK(in_channels <= MAX_IN_CHANNELS, "Input channels exceed maximum supported size");
    TORCH_CHECK(out_channels <= MAX_OUT_CHANNELS, "Output channels exceed maximum supported size");
    TORCH_CHECK(kernel_h <= MAX_KERNEL_SIZE && kernel_w <= MAX_KERNEL_SIZE, 
               "Kernel dimensions exceed maximum supported size");

    int pad_h = std::get<0>(padding);
    int pad_w = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    int height_out = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int width_out = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    // Copy weights to constant memory
    cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), 
                       out_channels * in_channels * kernel_h * kernel_w * sizeof(float));

    bool has_bias = bias.has_value();
    if (has_bias) {
        cudaMemcpyToSymbol(const_bias, bias->data_ptr<float>(), 
                          out_channels * sizeof(float));
    }

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());

    const int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (width_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (height_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size * ((out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK)
    );

    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_height,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        height_out,
        width_out,
        stride,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        has_bias
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Conv2D forward (CUDA)");
}