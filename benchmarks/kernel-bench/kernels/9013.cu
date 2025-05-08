#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for weights and bias
__constant__ float c_weight[1024];  // Adjust size based on expected maximum weight size
__constant__ float c_bias[256];     // Adjust size based on expected maximum bias size

__global__ void conv1d_const_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    int B,
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation,
    bool has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

    // Calculate indices
    int o = idx % out_size;
    int tmp = idx / out_size;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    float sum = 0.0f;
    int start_pos = o * stride;
    int end_pos = start_pos + (kernel_size - 1) * dilation;

    // Fast path when convolution window is fully within bounds
    if (end_pos < in_size) {
        for (int ic = 0; ic < in_channels; ++ic) {
            const float* x_ptr = x + b * (in_channels * in_size) + ic * in_size + start_pos;
            const int w_offset = oc * (in_channels * kernel_size) + ic * kernel_size;
            
            #pragma unroll 4
            for (int k = 0; k < kernel_size; ++k) {
                sum += x_ptr[k * dilation] * c_weight[w_offset + k];
            }
        }
    } else {
        // Boundary handling path
        for (int ic = 0; ic < in_channels; ++ic) {
            const float* x_ptr = x + b * (in_channels * in_size) + ic * in_size;
            const int w_offset = oc * (in_channels * kernel_size) + ic * kernel_size;
            
            #pragma unroll 4
            for (int k = 0; k < kernel_size; ++k) {
                int pos = start_pos + k * dilation;
                if (pos < in_size) {
                    sum += x_ptr[pos] * c_weight[w_offset + k];
                }
            }
        }
    }

    if (has_bias) {
        sum += c_bias[oc];
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

    // Check if weight size fits in constant memory
    TORCH_CHECK(weight.numel() <= 1024, "Weight tensor too large for constant memory");
    if (bias.has_value()) {
        TORCH_CHECK(bias->numel() <= 256, "Bias tensor too large for constant memory");
    }

    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size");

    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    // Copy weight and bias to constant memory
    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), 
                       weight.numel() * sizeof(float));
    
    bool has_bias = bias.has_value();
    if (has_bias) {
        cudaMemcpyToSymbol(c_bias, bias->data_ptr<float>(), 
                          bias->numel() * sizeof(float));
    }

    const float* x_data = x.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    int total_elements = B * out_channels * out_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    conv1d_const_kernel<<<blocks, threads>>>(
        x_data,
        output_data,
        B,
        in_channels,
        in_size,
        out_channels,
        kernel_size,
        out_size,
        stride,
        dilation,
        has_bias
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution forward with constant memory (CUDA)");
}