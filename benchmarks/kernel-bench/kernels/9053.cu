#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 3D grid indexing kernel for 1D convolution with dilation and stride
__global__ void conv1d_3d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
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
    // Map the 3D grid to output dimensions: (batch, out_channels, out_size)
    // blockIdx.z corresponds to batch index
    // blockIdx.y * blockDim.y + threadIdx.y corresponds to out_channel index
    // blockIdx.x * blockDim.x + threadIdx.x corresponds to spatial output index
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (o >= out_size || oc >= out_channels)
        return;

    float sum = 0.0f;

    // Iterate over input channels and apply convolution
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int input_pos = o * stride + k * dilation;
            if (input_pos < in_size) {
                int x_idx = b * (in_channels * in_size) + ic * in_size + input_pos;
                int w_idx = oc * (in_channels * kernel_size) + ic * kernel_size + k;
                sum += x[x_idx] * weight[w_idx];
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    int out_idx = b * (out_channels * out_size) + oc * out_size + o;
    output[out_idx] = sum;
}

// Forward function for the 1D convolution kernel using 3D grid indexing
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

    // Configure a 3D grid: use 16x16 threads per block for spatial and channel dimensions
    dim3 threads(32, 8, 1);  // Better warp alignment: 32 threads in x-dimension
    dim3 blocks(
        (out_size + threads.x - 1) / threads.x,
        (out_channels + threads.y - 1) / threads.y,
        B
    );

    conv1d_3d_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "1D convolution forward with 3D grid indexing (CUDA)");
}
