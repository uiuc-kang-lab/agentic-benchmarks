#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256  // Experiment with block sizes: 32, 64, 128, 256, 512
#endif

__global__ void conv1d_kernel_shared(
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

    // Calculate indices
    int o = idx % out_size;
    idx /= out_size;
    int oc = idx % out_channels;
    int b = idx / out_channels;

    float sum = 0.0f;
    int base_input_idx = o * stride;
    int b_offset = b * (in_channels * in_size);
    int oc_offset = oc * (in_channels * kernel_size);

    // Loop over input channels and kernel
    for (int ic = 0; ic < in_channels; ic++) {
        int x_offset = b_offset + ic * in_size;
        int w_offset = oc_offset + ic * kernel_size;
        for (int k = 0; k < kernel_size; k++) {
            int input_pos = base_input_idx + k * dilation;
            if (input_pos < in_size) {
                sum += x[x_offset + input_pos] * weight[w_offset + k];
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    int out_idx = b * (out_channels * out_size) + oc * out_size + o;
    output[out_idx] = sum;
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

    int B = x.size(0);
    int in_channels = x.size(1);
    int in_size = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size");

    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    int total_elements = B * out_channels * out_size;
    int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    conv1d_kernel<<<blocks, BLOCK_SIZE>>>(
        x_ptr, weight_ptr, bias_ptr, output_ptr,
        B, in_channels, in_size,
        out_channels, kernel_size, out_size,
        stride, dilation
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution forward (CUDA) with optimal block size tuning");
}
