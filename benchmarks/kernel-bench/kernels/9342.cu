#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
#define CHANNELS_PER_BLOCK 4

__global__ void conv2d_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
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
    int dilation_w) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int h_out = by * BLOCK_SIZE_Y + ty;
    int w_out = bx * BLOCK_SIZE_X + tx;
    int b = bz / ((out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK);
    int oc_start = (bz % ((out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK)) * CHANNELS_PER_BLOCK;

    if (h_out >= height_out || w_out >= width_out || b >= batch_size) return;

    float sums[CHANNELS_PER_BLOCK] = {0.0f};

    // Initialize with bias if present
    #pragma unroll
    for (int oc_offset = 0; oc_offset < CHANNELS_PER_BLOCK && oc_start + oc_offset < out_channels; ++oc_offset) {
        sums[oc_offset] = bias ? bias[oc_start + oc_offset] : 0.0f;
    }

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_in = h_out * stride + kh * dilation_h - pad_h;
            if (h_in >= 0 && h_in < input_height) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int w_in = w_out * stride + kw * dilation_w - pad_w;
                    if (w_in >= 0 && w_in < input_width) {
                        float x_val = x[b * in_channels * input_height * input_width +
                                      ic * input_height * input_width +
                                      h_in * input_width + w_in];

                        #pragma unroll
                        for (int oc_offset = 0; oc_offset < CHANNELS_PER_BLOCK && oc_start + oc_offset < out_channels; ++oc_offset) {
                            float w_val = weight[(oc_start + oc_offset) * in_channels * kernel_h * kernel_w +
                                               ic * kernel_h * kernel_w +
                                               kh * kernel_w + kw];
                            sums[oc_offset] += x_val * w_val;
                        }
                    }
                }
            }
        }
    }

    // Write results
    #pragma unroll
    for (int oc_offset = 0; oc_offset < CHANNELS_PER_BLOCK && oc_start + oc_offset < out_channels; ++oc_offset) {
        int out_idx = b * out_channels * height_out * width_out +
                      (oc_start + oc_offset) * height_out * width_out +
                      h_out * width_out + w_out;
        output[out_idx] = sums[oc_offset];
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

    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 blocks(
        (width_out + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (height_out + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        batch_size * ((out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK)
    );

    conv2d_kernel<<<blocks, threads>>>(
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