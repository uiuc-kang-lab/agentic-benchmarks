#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv1d_kernel_adaptive(
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
    // Calculate indices and the total number of output elements
    int output_channel_stride = gridDim.y * out_size;
    int idx = blockIdx.x * output_channel_stride + blockIdx.y * blockDim.x + threadIdx.x;
    int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

    // Convert flat index into multidimensional indices, ensuring correct and efficient mapping
    int o = blockIdx.y * blockDim.x + threadIdx.x;
    int b = idx / output_channel_stride;
    int oc = idx % output_channel_stride / out_size;

    float sum = 0.0f;
    int start_pos = o * stride;
    int end_pos = start_pos + (kernel_size - 1) * dilation;

    // More efficient boundary handling by leveraging thread block structure
    for (int ic = 0; ic < in_channels; ++ic) {
        const float* x_base = x + b * (in_channels * in_size) + ic * in_size;
        const float* w_base = weight + oc * (in_channels * kernel_size) + ic * kernel_size;

        if (end_pos < in_size) {
            #pragma unroll 4
            for (int k = 0; k < kernel_size; ++k) {
                sum += x_base[start_pos + k * dilation] * w_base[k];
            }
        } else {
            #pragma unroll 4
            for (int k = 0; k < kernel_size; ++k) {
                int input_pos = start_pos + k * dilation;
                sum += (input_pos < in_size) ? x_base[input_pos] * w_base[k] : 0.0f;
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

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

    int B = x.size(0);
    int in_channels = x.size(1);
    int in_size = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size");

    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = bias ? bias->data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    // Adjust block and grid dimensions for optimal execution on NVIDIA H100
    dim3 threads(256);
    dim3 blocks((B * out_channels * out_size + threads.x - 1) / threads.x, out_size);

    conv1d_kernel_adaptive<<<blocks, threads>>>(
        x_data,
        weight_data,
        bias_data,
        output_data,
        B,
        in_channels,
        in_size,
        out_channels,
        kernel_size,
        out_size,
        stride,
        dilation
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution forward (CUDA) with adaptive thread and block dimension optimization");
}