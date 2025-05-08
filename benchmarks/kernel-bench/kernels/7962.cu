#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define MAX_CONST_WEIGHT_ELEMENTS 16384

__constant__ float d_const_weight[MAX_CONST_WEIGHT_ELEMENTS];

__global__ void conv2d_kernel_coalesced(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias,
    int batch, int in_channels, int out_channels,
    int in_height, int in_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int out_height,
    int out_width
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int combo = blockIdx.z;
    int n = combo / out_channels;
    int oc = combo % out_channels;

    if(n < batch && oc < out_channels && oh < out_height && ow < out_width) {
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            int h_in = oh * stride - padding;
            int kh_start = (h_in < 0) ? ((-h_in + dilation - 1) / dilation) : 0;
            int kh_end = min(kernel_size, (in_height - h_in + dilation - 1) / dilation);
            for (int kh = kh_start; kh < kh_end; kh++) {
                int ih = h_in + kh * dilation;
                int w_in = ow * stride - padding;
                int kw_start = (w_in < 0) ? ((-w_in + dilation - 1) / dilation) : 0;
                int kw_end = min(kernel_size, (in_width - w_in + dilation - 1) / dilation);
                for (int kw = kw_start; kw < kw_end; kw++) {
                    int iw = w_in + kw * dilation;
                    int input_index = ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                    int weight_index = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_index] * d_const_weight[weight_index];
                }
            }
        }
        if(bias != nullptr) {
            sum += bias[oc];
        }
        int output_index = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
        output[output_index] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if(bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    TORCH_CHECK(groups == 1, "Only groups=1 is supported");

    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    TORCH_CHECK(weight.numel() <= MAX_CONST_WEIGHT_ELEMENTS, "Weight tensor too large for constant memory");

    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch, out_channels, out_height, out_width}, input.options());

    cudaMemcpyToSymbol(d_const_weight, weight.data_ptr<float>(), weight.numel() * sizeof(float));

    dim3 block(32, 32, 1);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        batch * out_channels
    );

    conv2d_kernel_coalesced<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        batch, in_channels, out_channels,
        in_height, in_width,
        kernel_size,
        stride, padding, dilation,
        out_height, out_width
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with coalesced memory access");
}
