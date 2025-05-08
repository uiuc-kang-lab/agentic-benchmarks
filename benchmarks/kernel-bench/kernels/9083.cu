#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline float compute_convolution(
    const float* x,
    const float* weight,
    int base_x,
    int base_w,
    int kernel_size,
    int stride,
    int dilation,
    int in_size,
    int o
) {
    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        int input_pos = o * stride + k * dilation;
        if (input_pos < in_size) {
            sum += __ldg(&x[base_x + input_pos]) * __ldg(&weight[base_w + k]);
        }
    }
    return sum;
}

__global__ void conv1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B,
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

    const int o = idx % out_size;
    const int oc = (idx / out_size) % out_channels;
    const int b = idx / (out_size * out_channels);

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        int base_x = b * (in_channels * in_size) + ic * in_size;
        int base_w = oc * (in_channels * kernel_size) + ic * kernel_size;
        sum += compute_convolution(x, weight, base_x, base_w, kernel_size, stride, dilation, in_size, o);
    }

    if (bias) sum += bias[oc];
    output[b * (out_channels * out_size) + oc * out_size + o] = sum;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int dilation
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
    TORCH_CHECK(weight.size(1) == x.size(1), "Input channels mismatch");

    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias->size(0) == weight.size(0), "Bias size mismatch");
    }

    const int B = x.size(0);
    const int in_channels = x.size(1);
    const int in_size = x.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    const int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size");

    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    const int total_elements = B * out_channels * out_size;
    constexpr int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    conv1d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, in_channels, in_size,
        out_channels, kernel_size, out_size,
        stride, dilation
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 1D convolution with modular device functions");
}