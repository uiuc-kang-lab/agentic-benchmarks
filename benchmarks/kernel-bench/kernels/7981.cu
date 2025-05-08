#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_coalesced_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
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

    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int c_out = blockIdx.z;

    if (out_x >= out_width || out_y >= out_height || c_out >= out_channels) return;

    float sum = 0.0f;

    const int out_x_origin = out_x * stride - padding;
    const int out_y_origin = out_y * stride - padding;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        const float* __restrict__ base_in = input + c_in * in_height * in_width;
        const float* __restrict__ base_weight = weight + (c_out * in_channels + c_in) * kernel_size * kernel_size;
        for (int ky = 0; ky < kernel_size; ++ky) {
            int in_y = out_y_origin + ky * dilation;
            if (in_y < 0 || in_y >= in_height) continue;
            int row_index = in_y * in_width;
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_x = out_x_origin + kx * dilation;
                if (in_x < 0 || in_x >= in_width) continue;
                sum += base_in[row_index + in_x] * base_weight[ky * kernel_size + kx];
            }
        }
    }

    float* __restrict__ out_ptr = output + (c_out * out_height + out_y) * out_width + out_x;
    *out_ptr = sum;
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

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(0);

    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    dim3 block(16, 8);  // 128 threads for better occupancy
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        out_channels);

    for (int b = 0; b < batch_size; ++b) {
        conv2d_coalesced_kernel<<<grid, block>>>(
            x[b].data_ptr<float>(),
            weight.data_ptr<float>(),
            output[b].data_ptr<float>(),
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            out_height,
            out_width);
    }

    if (bias.has_value()) {
        output += bias.value().view({1, out_channels, 1, 1});
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced memory 2D convolution");
}
