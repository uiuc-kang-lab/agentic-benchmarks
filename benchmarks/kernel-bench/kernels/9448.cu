#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 16
#define CHANNELS_PER_BLOCK 8

__global__ void conv2d_kernel_tuned(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int input_height,
    const int input_width,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int height_out,
    const int width_out,
    const int stride,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w) {

    

    const int w_out = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    const int h_out = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    const int oc_base = blockIdx.z * CHANNELS_PER_BLOCK;

    if (h_out >= height_out || w_out >= width_out) return;

    float sums[CHANNELS_PER_BLOCK] = {0.0f};
    
    #pragma unroll
    for (int oc_offset = 0; oc_offset < CHANNELS_PER_BLOCK && oc_base + oc_offset < out_channels; ++oc_offset) {
        sums[oc_offset] = bias ? bias[oc_base + oc_offset] : 0.0f;
    }

    for (int b = 0; b < batch_size; ++b) {
        for (int ic = 0; ic < in_channels; ++ic) {
            #pragma unroll 4
            for (int kh = 0; kh < kernel_h; ++kh) {
                const int h_in = h_out * stride + kh * dilation_h - pad_h;
                
                if (h_in >= 0 && h_in < input_height) {
                    #pragma unroll 4
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        const int w_in = w_out * stride + kw * dilation_w - pad_w;
                        
                        if (w_in >= 0 && w_in < input_width) {
                            const float x_val = x[b * in_channels * input_height * input_width +
                                               ic * input_height * input_width +
                                               h_in * input_width + w_in];
                            
                            #pragma unroll
                            for (int oc_offset = 0; oc_offset < CHANNELS_PER_BLOCK && oc_base + oc_offset < out_channels; ++oc_offset) {
                                const float w_val = weight[(oc_base + oc_offset) * in_channels * kernel_h * kernel_w +
                                                         ic * kernel_h * kernel_w +
                                                         kh * kernel_w + kw];
                                sums[oc_offset] = __fmaf_rn(x_val, w_val, sums[oc_offset]);
                            }
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int oc_offset = 0; oc_offset < CHANNELS_PER_BLOCK && oc_base + oc_offset < out_channels; ++oc_offset) {
            const int out_idx = b * out_channels * height_out * width_out +
                               (oc_base + oc_offset) * height_out * width_out +
                               h_out * width_out + w_out;
            output[out_idx] = sums[oc_offset];
            sums[oc_offset] = bias ? bias[oc_base + oc_offset] : 0.0f;
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

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        bias_ptr = bias->data_ptr<float>();
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int pad_h = std::get<0>(padding);
    int pad_w = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    int height_out = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int width_out = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());

    if (height_out == 0 || width_out == 0) return output;

    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 blocks(
        (width_out + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (height_out + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK
    );

    conv2d_kernel_tuned<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
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
        dilation_w
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Conv2D forward (CUDA)");
}