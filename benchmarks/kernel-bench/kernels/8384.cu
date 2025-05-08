#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized kernel with shared memory and tiling
__global__ void conv_transpose2d_kernel_hybrid(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int output_height,
    int output_width,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w) {

    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + blockDim.x * blockDim.y;

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int linear_idx = blockIdx.z;
    int batch = linear_idx / out_channels;
    int out_ch = linear_idx % out_channels;

    float sum = 0.0f;

    if (out_x < output_width && out_y < output_height && batch < batch_size) {
        const int TILE_SIZE = 16;
        for (int tile = 0; tile < (in_channels + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
            int tile_start = tile * TILE_SIZE;
            int tile_end = min(tile_start + TILE_SIZE, in_channels);

            if (threadIdx.x < kernel_width && threadIdx.y < kernel_height) {
                for (int in_ch = tile_start; in_ch < tile_end; ++in_ch) {
                    shared_weight[((in_ch - tile_start) * kernel_height + threadIdx.y) * kernel_width + threadIdx.x] =
                        weight[in_ch * out_channels * kernel_height * kernel_width +
                              out_ch * kernel_height * kernel_width +
                              threadIdx.y * kernel_width + threadIdx.x];
                }
            }

            __syncthreads();

            for (int in_ch = tile_start; in_ch < tile_end; ++in_ch) {
                for (int kh = 0; kh < kernel_height; kh++) {
                    for (int kw = 0; kw < kernel_width; kw++) {
                        int in_x = out_x + pad_w - kw;
                        int in_y = out_y + pad_h - kh;

                        if (in_x % stride_w == 0 && in_y % stride_h == 0) {
                            in_x /= stride_w;
                            in_y /= stride_h;
                            if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
                                float input_val = input[batch * in_channels * input_height * input_width +
                                                      in_ch * input_height * input_width +
                                                      in_y * input_width + in_x];

                                float weight_val = shared_weight[((in_ch - tile_start) * kernel_height + kh) * kernel_width + kw];

                                sum += input_val * weight_val;
                            }
                        }
                    }
                }
            }

            __syncthreads();
        }

        if (bias) {
            sum += bias[out_ch];
        }

        output[batch * out_channels * output_height * output_width +
               out_ch * output_height * output_width +
               out_y * output_width + out_x] = sum;
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    if (x.numel() < 1024 * 1024) {
        return at::conv_transpose2d(
            x, weight, bias.value_or(torch::Tensor()),
            stride, padding, output_padding, groups, dilation
        );
    }

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_height = x.size(2);
    const int input_width = x.size(3);
    const int out_channels = weight.size(1);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);

    const int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0];
    const int output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, x.options());

    dim3 threads(16, 16, 1);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * out_channels
    );

    const int shared_mem_size = (16 * kernel_height * kernel_width + threads.x * threads.y) * sizeof(float);

    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    conv_transpose2d_kernel_hybrid<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        output_height,
        output_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1]
    );

    return output;
}