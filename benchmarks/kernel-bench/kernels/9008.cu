#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv1d_kernel_strided(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int B,
    const int in_channels,
    const int in_size,
    const int out_channels,
    const int kernel_size,
    const int out_size,
    const int stride,
    const int dilation
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int total_elements = B * out_channels * out_size;

    for (int idx = tid; idx < total_elements; idx += total_threads) {
        const int o = idx % out_size;
        const int tmp = idx / out_size;
        const int oc = tmp % out_channels;
        const int b = tmp / out_channels;

        const int start_pos = o * stride;
        const int end_pos = start_pos + (kernel_size - 1) * dilation;

        float sum = 0.0f;
        
        if (end_pos < in_size) {
            #pragma unroll 4
            for (int ic = 0; ic < in_channels; ++ic) {
                const float* x_base = x + b * (in_channels * in_size) + ic * in_size + start_pos;
                const float* w_base = weight + oc * (in_channels * kernel_size) + ic * kernel_size;
                
                #pragma unroll
                for (int k = 0; k < kernel_size; ++k) {
                    sum += x_base[k * dilation] * w_base[k];
                }
            }
        } else {
            #pragma unroll 4
            for (int ic = 0; ic < in_channels; ++ic) {
                const float* x_base = x + b * (in_channels * in_size) + ic * in_size;
                const float* w_base = weight + oc * (in_channels * kernel_size) + ic * kernel_size;
                
                for (int k = 0; k < kernel_size; ++k) {
                    const int pos = start_pos + k * dilation;
                    if (pos < in_size) {
                        sum += x_base[pos] * w_base[k];
                    }
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
    const float* bias_data = bias.has_value() ? bias->data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    const int threads = 256;
    const int max_blocks = 65535;
    const int min_threads_per_block = 32;
    const int total_elements = B * out_channels * out_size;
    
    int blocks = (total_elements + threads - 1) / threads;
    blocks = min(blocks, max_blocks);
    blocks = max(blocks, min_threads_per_block);

    conv1d_kernel_strided<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Strided 1D convolution forward (CUDA)");
}