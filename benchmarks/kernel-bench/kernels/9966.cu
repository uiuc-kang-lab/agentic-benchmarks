#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_HEIGHT 4
#define BLOCK_WIDTH 32

__global__ void depthwise_conv2d_coalesced(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int channels_per_group
) {
    // Thread coordinates within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Output coordinates
    const int x_out = blockIdx.x * BLOCK_WIDTH + tx;
    const int y_out = blockIdx.y * BLOCK_HEIGHT + ty;
    const int b = blockIdx.z / out_channels;
    const int c_out = blockIdx.z % out_channels;

    if (x_out >= out_w || y_out >= out_h || b >= batch_size) return;

    const int g = c_out / channels_per_group;
    const int m = c_out % channels_per_group;

    // Pre-compute input batch and channel offset for coalesced access
    const int in_batch_offset = b * in_channels * in_h * in_w;
    const int in_channel_offset = g * in_h * in_w;
    const int weight_offset = (g * channels_per_group + m) * kernel_h * kernel_w;

    float sum = 0.0f;

    // Compute convolution with coalesced memory access pattern
    #pragma unroll 3
    for (int kh = 0; kh < kernel_h; ++kh) {
        const int h_in = y_out * stride_h - padding_h + kh * dilation_h;
        
        if (h_in >= 0 && h_in < in_h) {
            const int in_h_offset = h_in * in_w;
            
            #pragma unroll 3
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int w_in = x_out * stride_w - padding_w + kw * dilation_w;
                
                if (w_in >= 0 && w_in < in_w) {
                    // Coalesced input access - consecutive threads read consecutive locations
                    const int in_idx = in_batch_offset + in_channel_offset + in_h_offset + w_in;
                    const int w_idx = weight_offset + kh * kernel_w + kw;
                    
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    // Coalesced output write - consecutive threads write to consecutive locations
    const int out_idx = ((b * out_channels + c_out) * out_h + y_out) * out_w + x_out;
    output[out_idx] = sum;
}

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

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels = groups * weight.size(1);
    const int channels_per_group = out_channels / groups;
    const int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());
    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    // Configure grid and block dimensions for coalesced access
    dim3 threads(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 blocks(
        (out_w + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
        (out_h + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT,
        batch_size * out_channels
    );

    depthwise_conv2d_coalesced<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
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
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Depthwise Conv2D forward (CUDA)");
}