#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define CHANNELS_PER_BLOCK 8

template<int KERNEL_SIZE>
__device__ __forceinline__ float compute_single_output(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    int b, int ic, int h_out, int w_out,
    int input_height, int input_width,
    int stride, int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int oc, int in_channels) {
    
    float sum = 0.0f;
    
    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
        int h_in = h_out * stride + kh * dilation_h - pad_h;
        if (h_in >= 0 && h_in < input_height) {
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                int w_in = w_out * stride + kw * dilation_w - pad_w;
                if (w_in >= 0 && w_in < input_width) {
                    float x_val = x[b * in_channels * input_height * input_width +
                                  ic * input_height * input_width +
                                  h_in * input_width + w_in];
                    float w_val = weight[oc * in_channels * KERNEL_SIZE * KERNEL_SIZE +
                                       ic * KERNEL_SIZE * KERNEL_SIZE +
                                       kh * KERNEL_SIZE + kw];
                    sum += x_val * w_val;
                }
            }
        }
    }
    return sum;
}

template<int KERNEL_SIZE>
__device__ __forceinline__ void process_channel_block(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* sums,
    int b, int ic, int h_out, int w_out,
    int input_height, int input_width,
    int stride, int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int oc_start, int out_channels, int in_channels) {
    
    #pragma unroll
    for (int oc_offset = 0; oc_offset < CHANNELS_PER_BLOCK; ++oc_offset) {
        if (oc_start + oc_offset < out_channels) {
            sums[oc_offset] += compute_single_output<KERNEL_SIZE>(
                x, weight, b, ic, h_out, w_out,
                input_height, input_width,
                stride, pad_h, pad_w,
                dilation_h, dilation_w,
                oc_start + oc_offset, in_channels);
        }
    }
}

template<int KERNEL_SIZE>
__global__ void conv2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int out_channels,
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

    // Initialize with bias
    #pragma unroll
    for (int oc_offset = 0; oc_offset < CHANNELS_PER_BLOCK && oc_start + oc_offset < out_channels; ++oc_offset) {
        sums[oc_offset] = bias ? bias[oc_start + oc_offset] : 0.0f;
    }

    // Process input channels
    #pragma unroll 4
    for (int ic = 0; ic < in_channels; ++ic) {
        process_channel_block<KERNEL_SIZE>(
            x, weight, sums,
            b, ic, h_out, w_out,
            input_height, input_width,
            stride, pad_h, pad_w,
            dilation_h, dilation_w,
            oc_start, out_channels, in_channels);
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
    int kernel_size = weight.size(2);

    int pad_h = std::get<0>(padding);
    int pad_w = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    int height_out = (input_height + 2 * pad_h - dilation_h * (kernel_size - 1) - 1) / stride + 1;
    int width_out = (input_width + 2 * pad_w - dilation_w * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());

    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 blocks(
        (width_out + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (height_out + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        batch_size * ((out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK)
    );

    if (kernel_size == 3) {
        conv2d_kernel<3><<<blocks, threads>>>(
            x.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
            output.data_ptr<float>(), batch_size, in_channels,
            input_height, input_width, out_channels,
            height_out, width_out, stride,
            pad_h, pad_w, dilation_h, dilation_w);
    } else if (kernel_size == 5) {
        conv2d_kernel<5><<<blocks, threads>>>(
            x.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
            output.data_ptr<float>(), batch_size, in_channels,
            input_height, input_width, out_channels,
            height_out, width_out, stride,
            pad_h, pad_w, dilation_h, dilation_w);
    } else {
        conv2d_kernel<7><<<blocks, threads>>>(
            x.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
            output.data_ptr<float>(), batch_size, in_channels,
            input_height, input_width, out_channels,
            height_out, width_out, stride,
            pad_h, pad_w, dilation_h, dilation_w);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Conv2D forward (CUDA)");
}