#include <torch/extension.h>

__global__ void conv_transpose2d_kernel(
float* output,
const float* input,
const float* weight,
const float* bias,
int batch_size,
int in_channels,
int out_channels,
int input_height,
int input_width,
int kernel_size,
int stride,
int padding,
int output_padding) {

    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_input + (kernel_size * kernel_size);

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    int out_y = by * blockDim.y + ty;
    int out_x = bx * blockDim.x + tx;

    if (out_y >= output_height || out_x >= output_width)
        return;

    for (int n = 0; n < batch_size; n++) {
        for (int oc = 0; oc < out_channels; oc++) {
            float sum = bias ? bias[oc] : 0.0f;

            #pragma unroll 4
            for (int ic = 0; ic < in_channels; ic++) {
                #pragma unroll
                for (int kh = 0; kh < kernel_size; kh++) {
                    #pragma unroll
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int in_y = (out_y + padding - kh) / stride;
                        int in_x = (out_x + padding - kw) / stride;

                        if (in_y >= 0 && in_y < input_height &&
                            in_x >= 0 && in_x < input_width &&
                            (out_y + padding - kh) % stride == 0 &&
                            (out_x + padding - kw) % stride == 0) {

                            int in_idx = ((n * in_channels + ic) * input_height + in_y) * input_width + in_x;
                            int w_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;

                            sum += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }

            int out_idx = ((n * out_channels + oc) * output_height + out_y) * output_width + out_x;
            output[out_idx] = sum;
        }
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    auto input = x;
    auto kernel_size = weight.size(2);
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto input_height = input.size(2);
    auto input_width = input.size(3);

    auto output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    auto output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width},
                              input.options());

    dim3 threads(16, 16);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        1
    );

    int shared_mem_size = (kernel_size * kernel_size + out_channels) * sizeof(float);

    conv_transpose2d_kernel<<<blocks, threads, shared_mem_size>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        output_padding
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA)");
}