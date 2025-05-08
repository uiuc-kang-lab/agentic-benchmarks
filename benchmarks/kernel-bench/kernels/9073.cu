#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel maps the output tensor using a 3D grid and 2D blocks:
//   - blockIdx.z corresponds to the batch index (B)
//   - blockIdx.y corresponds to the output channel index (OC)
//   - blockIdx.x corresponds to the tile of the output spatial index (O)
// Each thread computes one output element for a specific (b, oc, o) coordinate.

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
    // Map thread indices to output tensor coordinates
    int o = blockIdx.x * blockDim.x + threadIdx.x;      // output spatial index
    int oc = blockIdx.y * blockDim.y + threadIdx.y;       // output channel
    int b = blockIdx.z;                                   // batch index

    // Check bounds
    if (o >= out_size || oc >= out_channels) return;

    float sum = 0.0f;
    
    // Loop over all input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        int base_x = b * (in_channels * in_size) + ic * in_size;
        int base_w = oc * (in_channels * kernel_size) + ic * kernel_size;
        
        // Loop over the kernel elements
        for (int k = 0; k < kernel_size; ++k) {
            int input_pos = o * stride + k * dilation;
            if (input_pos < in_size) {
                sum += x[base_x + input_pos] * weight[base_w + k];
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

    // Define 2D block dimensions and 3D grid dimensions
    // Using blockDim (16, 16) gives 256 threads per block, which is effective for many architectures.
    dim3 blockDim(16, 16);
    dim3 gridDim((out_size + blockDim.x - 1) / blockDim.x,
                 (out_channels + blockDim.y - 1) / blockDim.y,
                 B);

    conv1d_kernel<<<gridDim, blockDim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
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
    m.def("forward", &forward, "1D convolution forward with 3D grid kernel (CUDA)");
}
