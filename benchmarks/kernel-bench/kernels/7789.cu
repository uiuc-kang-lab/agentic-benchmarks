#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_ldg_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // may be nullptr if not provided
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_height,
    int kernel_width,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation,
    int groups) {

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;

    if (w < out_width && h < out_height && oc < out_channels) {
      for (int b = 0; b < batch_size; ++b) {
        float sum = 0.0f;
        int group_out_channels = out_channels / groups;
        int group = oc / group_out_channels;
        int in_channels_per_group = in_channels / groups;

        for (int c = 0; c < in_channels_per_group; ++c) {
          int input_channel = group * in_channels_per_group + c;
          for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
              int in_y = h * stride - padding + kh * dilation;
              int in_x = w * stride - padding + kw * dilation;
              if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                int input_idx = ((b * in_channels + input_channel) * in_height + in_y) * in_width + in_x;
                int weight_idx = (((oc * in_channels_per_group + c) * kernel_height) + kh) * kernel_width + kw;
                sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
              }
            }
          }
        }
        if (bias != nullptr) {
          sum += __ldg(&bias[oc]);
        }
        int output_idx = ((b * out_channels + oc) * out_height + h) * out_width + w;
        output[output_idx] = sum;
      }
    }
}

// The forward function sets up dimensions and launches the kernel with read-only cache optimizations
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
    if (bias.has_value()) {
      CHECK_INPUT(bias.value());
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto options = x.options();
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, options);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
      bias_ptr = bias.value().data_ptr<float>();
    }

    dim3 block_size(32, 16, 1);
    dim3 grid_size(
        (out_width + block_size.x - 1) / block_size.x,
        (out_height + block_size.y - 1) / block_size.y,
        out_channels  // one block per output channel
    );

    conv2d_ldg_kernel<<<grid_size, block_size>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride,
        padding,
        dilation,
        groups
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution with __ldg Optimizations");
}