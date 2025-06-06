#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv1d_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int B,
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_step = blockDim.x * gridDim.x;
    int total_elements = B * out_channels * out_size;

    // Each thread processes multiple elements with stride_step
    for (int idx = tid; idx < total_elements; idx += stride_step) {
        int o = idx % out_size;
        int tmp = idx / out_size;
        int oc = tmp % out_channels;
        int b = tmp / out_channels;

        float sum = 0.0f;

        // Pre-calculate base indices for input and weight
        int x_base = b * (in_channels * in_size);
        int w_base = oc * (in_channels * kernel_size);

        #pragma unroll
        for (int ic = 0; ic < in_channels; ++ic) {
            int x_offset = x_base + ic * in_size;
            int w_offset = w_base + ic * kernel_size;

            #pragma unroll
            for (int k = 0; k < kernel_size; ++k) {
                int input_pos = o * stride + k * dilation;
                if (input_pos >= 0 && input_pos < in_size) {
                    sum += x[x_offset + input_pos] * weight[w_offset + k];
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[oc];
        }
        
        output[b * (out_channels * out_size) + oc * out_size + o] = sum;
    }
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

    int threads = 256;
    int blocks = std::min(65535, (output.numel() + threads - 1) / threads);

    conv1d_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "1D convolution forward (CUDA)");
}