#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define maximum weight size for constant memory (in floats). Adjust if needed.
#define MAX_WEIGHT_SIZE 16384

// Declare constant memory for the weight, which is read-only and frequently accessed
__constant__ float const_weight[MAX_WEIGHT_SIZE];

// CUDA kernel for transposed 2D convolution using constant memory for weight
// Assumes square input and square kernel
// Input x shape: [N, in_channels, in_size, in_size]
// Weight shape: [in_channels, out_channels_per_group, kernel_size, kernel_size]
// groups: number of groups; out_channels = out_channels_per_group * groups

__global__ void conv_transpose2d_cuda_kernel(
    const float* __restrict__ input,
    const int N,
    const int in_channels,
    const int in_size,              // spatial dimension of input (square)
    const float* bias,              // may be nullptr
    float* output,
    const int out_channels,
    const int out_size,             // spatial dimension of output (square)
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int in_channels_per_group,
    const int out_channels_per_group) {

    // Each thread computes one output element: index corresponds to (n, out_channel, out_y, out_x)
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_channels * out_size * out_size;
    if (index >= total) return;

    // Compute spatial and channel indices from linear index (NCHW layout)
    int out_x = index % out_size;
    int tmp = index / out_size;
    int out_y = tmp % out_size;
    tmp = tmp / out_size;
    int out_c = tmp % out_channels;
    int n = tmp / out_channels;

    float sum = 0.0f;

    // Determine the group and relative output channel in that group
    int group = out_c / out_channels_per_group;
    int weight_out_channel = out_c - group * out_channels_per_group;  // relative index in group's output channels

    // Loop over the input channels for this group
    for (int in_c_local = 0; in_c_local < in_channels_per_group; in_c_local++) {
        int actual_in_c = group * in_channels_per_group + in_c_local;
        
        // Loop over kernel spatial dimensions
        for (int k_y = 0; k_y < kernel_size; k_y++) {
            for (int k_x = 0; k_x < kernel_size; k_x++) {
                // Compute the corresponding input coordinates based on transposed convolution formula
                // out_y = in_y * stride - padding + k_y  =>  in_y = (out_y + padding - k_y) / stride
                int numerator_y = out_y + padding - k_y;
                int numerator_x = out_x + padding - k_x;
                
                // Only proceed if the division by stride is exact
                if (numerator_y % stride == 0 && numerator_x % stride == 0) {
                    int in_y = numerator_y / stride;
                    int in_x = numerator_x / stride;
                    
                    // Check bounds of input spatial dimensions
                    if (in_y >= 0 && in_y < in_size && in_x >= 0 && in_x < in_size) {
                        int input_idx = ((n * in_channels + actual_in_c) * in_size + in_y) * in_size + in_x;
                        
                        // Compute weight index using full input channel index
                        // Weight shape: [in_channels, out_channels_per_group, kernel_size, kernel_size]
                        int weight_idx = (((actual_in_c) * out_channels_per_group + weight_out_channel) * kernel_size + k_y) * kernel_size + k_x;
                        
                        sum += input[input_idx] * const_weight[weight_idx];
                    }
                }
            }
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[out_c];
    }

    // Write result to output tensor; output shape: [N, out_channels, out_size, out_size]
    int output_idx = ((n * out_channels + out_c) * out_size + out_y) * out_size + out_x;
    output[output_idx] = sum;
}

// Forward function that sets up constant memory for the weight and launches the CUDA kernel

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    float* bias_ptr = nullptr;
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
        bias_ptr = bias.value().data_ptr<float>();
    }

    // Input dimensions: x shape [N, in_channels, in_size, in_size] (square input assumed)
    int N = x.size(0);
    int in_channels = x.size(1);
    int in_size = x.size(2);

    // Weight dimensions: expected shape [in_channels, out_channels_per_group, kernel_size, kernel_size]
    int kernel_size = weight.size(2);
    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;

    // Compute output spatial size using the transposed convolution formula
    int out_size = (in_size - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Create output tensor
    auto output = torch::zeros({N, out_channels, out_size, out_size}, x.options());

    // Determine channels per group
    int in_channels_per_group = in_channels / groups;

    // Ensure the weight fits in constant memory
    int weight_numel = weight.numel();
    TORCH_CHECK(weight_numel <= MAX_WEIGHT_SIZE,
                "Weight tensor is too large for constant memory. Current numel: ", weight_numel,
                ", maximum allowed: ", MAX_WEIGHT_SIZE);

    // Copy weight to constant memory. Use cudaMemcpyDeviceToDevice since weight is already on CUDA.
    cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), weight_numel * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    // Launch the CUDA kernel
    int total_threads = N * out_channels * out_size * out_size;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    conv_transpose2d_cuda_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        N,
        in_channels,
        in_size,
        bias_ptr,
        output.data_ptr<float>(),
        out_channels,
        out_size,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        in_channels_per_group,
        out_channels_per_group
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward with constant memory (CUDA)");
}
