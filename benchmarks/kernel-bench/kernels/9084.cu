#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using 2D grid: each block in y-dimension corresponds to one (batch, out_channel) pair.
// Threads in x-dimension compute consecutive output positions, ensuring coalesced global memory accesses for output.

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
    // Compute the output position index within the tile
    int out_tile = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each block in y-dimension corresponds to a unique (b, oc) pair
    int combined = blockIdx.y;  // combined index for batch and output channel
    int b = combined / out_channels;
    int oc = combined % out_channels;

    if (out_tile >= out_size) return;

    // Compute output index in flattened output tensor (B, out_channels, out_size) in row-major order
    int out_index = b * (out_channels * out_size) + oc * out_size + out_tile;
    float sum = 0.0f;

    // Starting input position corresponding to this output index
    int o_input = out_tile * stride;

    // Loop over input channels and kernel elements
    for (int ic = 0; ic < in_channels; ++ic) {
        // Calculate base offsets for input and weight
        int input_base = b * (in_channels * in_size) + ic * in_size;
        int weight_base = oc * (in_channels * kernel_size) + ic * kernel_size;
        
        // Unroll kernel loop if possible
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            int input_idx = o_input + k * dilation;
            if (input_idx < in_size) {
                sum += x[input_base + input_idx] * weight[weight_base + k];
            }
        }
    }

    // Add bias if provided
    if (bias) {
        sum += bias[oc];
    }
    
    // Write output; adjacent threads write to contiguous memory locations along the out_size dimension
    output[out_index] = sum;
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

    // Calculate output size analogous to PyTorch's conv1d formula
    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size");

    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    // Configure 2D grid: x-dimension covers the output positions; y-dimension covers (batch, out_channel) pairs.
    int block_x = 128;
    dim3 block(block_x);
    dim3 grid((out_size + block_x - 1) / block_x, B * out_channels);

    conv1d_kernel<<<grid, block>>>(
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
    m.def("forward", &forward, "Optimized 1D convolution forward with coalesced memory accesses (CUDA)");
}
