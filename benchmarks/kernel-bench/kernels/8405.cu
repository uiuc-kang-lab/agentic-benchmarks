#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS 256
#define SMALL_KERNEL_THRESHOLD 5
#define SMALL_CHANNEL_THRESHOLD 64

__global__ void conv_transpose2d_shared_memory_kernel(
    const float * __restrict__ input,
    const float * __restrict__ weight,
    const float * __restrict__ bias,
    float * __restrict__ output,
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
    float* shared_weight = shared_mem + in_channels * kernel_height * kernel_width;

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int out_x = bid % output_width;
    int out_y = (bid / output_width) % output_height;
    int out_ch = (bid / (output_width * output_height)) % out_channels;
    int batch = bid / (output_width * output_height * out_channels);

    if (tid < in_channels * kernel_height * kernel_width) {
        int ch = tid / (kernel_height * kernel_width);
        int kh = (tid / kernel_width) % kernel_height;
        int kw = tid % kernel_width;
        shared_weight[tid] = weight[ch * out_channels * kernel_height * kernel_width + out_ch * kernel_height * kernel_width + kh * kernel_width + kw];
    }

    __syncthreads();

    float sum = 0.0f;
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int kh = 0; kh < kernel_height; kh++) {
            for (int kw = 0; kw < kernel_width; kw++) {
                int in_x = out_x + pad_w - kw;
                int in_y = out_y + pad_h - kh;
                if (in_x % stride_w == 0 && in_y % stride_h == 0) {
                    in_x /= stride_w;
                    in_y /= stride_h;
                    if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
                        if (tid < in_channels * input_height * input_width) {
                            shared_input[tid] = input[batch * in_channels * input_height * input_width + in_ch * input_height * input_width + in_y * input_width + in_x];
                        }
                        __syncthreads();
                        sum += shared_input[in_ch * input_height * input_width + in_y * input_width + in_x] * shared_weight[in_ch * kernel_height * kernel_width + kh * kernel_width + kw];
                    }
                }
            }
        }
    }

    if (bias) {
        sum += bias[out_ch];
    }

    output[batch * out_channels * output_height * output_width + out_ch * output_height * output_width + out_y * output_width + out_x] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(1);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0];
    int output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    int shared_mem_size = in_channels * kernel_height * kernel_width * sizeof(float) * 2;

    int num_output_pixels = batch_size * out_channels * output_height * output_width;

    dim3 blocks(num_output_pixels);
    dim3 threads(THREADS);

    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    conv_transpose2d_shared_memory_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "Shared Memory ConvTranspose2D forward (CUDA)");
}