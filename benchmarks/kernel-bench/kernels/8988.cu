#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized 1D convolution kernel combining pointer arithmetic, loop unrolling, and minimal divergence
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

    // Compute output index with reduced divergence
    int o = idx % out_size;
    int tmp = idx / out_size;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    float sum = 0.0f;
    int start_pos = o * stride;
    int end_pos = start_pos + (kernel_size - 1) * dilation;

    // If the entire convolution window fits within the input, we can eliminate per-step boundary checks
    if (end_pos < in_size) {
        for (int ic = 0; ic < in_channels; ++ic) {
            // Compute base pointers using pre-calculated offsets
            const float* __restrict__ x_ptr = x + b * in_channels * in_size + ic * in_size + start_pos;
            const float* __restrict__ w_ptr = weight + oc * in_channels * kernel_size + ic * kernel_size;
            #pragma unroll
            for (int k = 0; k < kernel_size; ++k) {
                sum += x_ptr[k * dilation] * w_ptr[k];
            }
        }
    } else {
        // For positions where part of the window may be out-of-bound, check each position
        for (int ic = 0; ic < in_channels; ++ic) {
            const float* __restrict__ x_ptr = x + b * in_channels * in_size + ic * in_size;
            const float* __restrict__ w_ptr = weight + oc * in_channels * kernel_size + ic * kernel_size;
            #pragma unroll
            for (int k = 0; k < kernel_size; ++k) {
                int pos = start_pos + k * dilation;
                if (pos < in_size)
                    sum += x_ptr[pos] * w_ptr[k];
            }
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[oc];
    }

    // Write the result to the output tensor
    output[b * (out_channels * out_size) + oc * out_size + o] = sum;
}

// Forward function wrapped with PyBind11
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
        TORCH_CHECK(bias.value().device().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias.value().size(0) == weight.size(0), "Bias size mismatch");
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
    const float* bias_data = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    int total_elements = B * out_channels * out_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

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
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 1D convolution forward (CUDA)");
}
