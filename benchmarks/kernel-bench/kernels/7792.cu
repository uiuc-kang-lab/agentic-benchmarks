#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device function to load input value with bounds checking
__device__ __forceinline__ float load_input(
    const float* input,
    int b, int ic, int h, int w,
    int in_height, int in_width, int in_channels) {
    if (h >= 0 && h < in_height && w >= 0 && w < in_width) {
        return input[((b * in_channels + ic) * in_height + h) * in_width + w];
    }
    return 0.0f;
}

// Device function to load weight value
__device__ __forceinline__ float load_weight(
    const float* weight,
    int oc, int ic, int kh, int kw,
    int kernel_width, int in_channels_per_group) {
    return weight[(((oc * in_channels_per_group + ic) * kh) * kernel_width) + kw];
}

// Device function to compute single convolution window
__device__ __forceinline__ float compute_window(
    const float* input,
    const float* weight,
    int b, int oc, int h, int w,
    int in_channels_per_group,
    int kernel_height, int kernel_width,
    int stride, int padding, int dilation,
    int in_height, int in_width, int in_channels,
    int group_offset) {
    
    float sum = 0.0f;
    #pragma unroll 3
    for (int ic = 0; ic < in_channels_per_group; ++ic) {
        #pragma unroll
        for (int kh = 0; kh < kernel_height; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < kernel_width; ++kw) {
                int in_h = h * stride - padding + kh * dilation;
                int in_w = w * stride - padding + kw * dilation;
                float in_val = load_input(input, b, group_offset + ic, in_h, in_w, in_height, in_width, in_channels);
                float weight_val = load_weight(weight, oc, ic, kh, kw, kernel_width, in_channels_per_group);
                sum += in_val * weight_val;
            }
        }
    }
    return sum;
}

__global__ void conv2d_modular_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
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

    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int oc = blockIdx.z;

    if (w >= out_width || h >= out_height || oc >= out_channels) return;

    const int group_out_channels = out_channels / groups;
    const int group = oc / group_out_channels;
    const int in_channels_per_group = in_channels / groups;
    const int group_offset = group * in_channels_per_group;

    #pragma unroll 4
    for (int b = 0; b < batch_size; ++b) {
        float result = compute_window(
            input, weight,
            b, oc, h, w,
            in_channels_per_group,
            kernel_height, kernel_width,
            stride, padding, dilation,
            in_height, in_width, in_channels,
            group_offset
        );

        if (bias != nullptr) {
            result += bias[oc];
        }

        output[((b * out_channels + oc) * out_height + h) * out_width + w] = result;
    }
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
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);

    const int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, x.options());

    dim3 block_size(32, 16);
    dim3 grid_size(
        (out_width + block_size.x - 1) / block_size.x,
        (out_height + block_size.y - 1) / block_size.y,
        out_channels
    );

    conv2d_modular_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
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
    m.def("forward", &forward, "CUDA 2D Convolution with Modular Device Functions");
}