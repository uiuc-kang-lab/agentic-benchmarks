#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ float constant_weights[32768]; // 128KB buffer for H100's constant cache (max kernel size ~32k elements)

__global__ void conv1d_kernel(
    const float* __restrict__ x,
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

    int o = idx % out_size;
    int oc = (idx / out_size) % out_channels;
    int b = idx / (out_size * out_channels);

    float sum = 0.0f;
    const int weight_stride = in_channels * kernel_size;

    for (int ic = 0; ic < in_channels; ++ic) {
        const int input_base = b * in_channels * in_size + ic * in_size;
        const int weight_base = oc * weight_stride + ic * kernel_size;

        for (int k = 0; k < kernel_size; ++k) {
            int input_pos = o * stride + k * dilation;
            if (input_pos >= 0 && input_pos < in_size) {
                sum += x[input_base + input_pos] * constant_weights[weight_base + k];
            }
        }
    }

    if (bias != nullptr) sum += bias[oc];
    output[b * out_channels * out_size + oc * out_size + o] = sum;
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

    const size_t weight_bytes = weight.nbytes();
    TORCH_CHECK(weight_bytes <= sizeof(constant_weights),
              ("Weight size exceeds constant memory capacity (" + std::to_string(sizeof(constant_weights)) + " bytes) - try reducing kernel size").c_str());

    int B = x.size(0);
    int in_channels = x.size(1);
    int in_size = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size");

    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    // Copy weights to constant memory
    cudaMemcpyToSymbol(constant_weights, weight.data_ptr<float>(), weight_bytes);

    const float* x_data = x.data_ptr<float>();
    const float* bias_data = bias ? bias->data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    int total_elements = B * out_channels * out_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    conv1d_kernel<<<blocks, threads>>>(x_data, bias_data, output_data,
        B, in_channels, in_size, out_channels, kernel_size,
        out_size, stride, dilation);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution with constant memory optimization (CUDA)");
}