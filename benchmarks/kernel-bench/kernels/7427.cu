#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cassert>

// Kernel to compute each output element for transposed convolution.
// Input shape: [N, C_in, H_in, W_in]
// Weight shape: [C_in, C_out, K, K]
// Output shape: [N, C_out, H_out, W_out]
__global__ void conv_transpose2d_kernel_output(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K,
    int stride,
    int padding,
    int H_out, int W_out) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (index >= total) return;

    // Decode linear index to (n, oc, h_out, w_out)
    int w_out = index % W_out;
    int tmp = index / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int oc = tmp % C_out;
    int n = tmp / C_out;

    float sum = 0.f;
    // Loop over input channels and kernel window
    for (int c = 0; c < C_in; ++c) {
        for (int ki = 0; ki < K; ++ki) {
            for (int kj = 0; kj < K; ++kj) {
                int h_in_candidate = h_out + padding - ki;
                int w_in_candidate = w_out + padding - kj;
                // Check if candidate indices align with stride
                if (h_in_candidate % stride != 0 || w_in_candidate % stride != 0)
                    continue;
                int h_in = h_in_candidate / stride;
                int w_in = w_in_candidate / stride;
                // Validate input bounds
                if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in)
                    continue;
                int input_idx = n * (C_in * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in;
                int weight_idx = c * (C_out * K * K) + oc * (K * K) + ki * K + kj;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    output[index] = sum;
}

// Kernel to add bias to each output element
__global__ void add_bias_kernel(
    float* output,
    const float* bias,
    int total,
    int C_out,
    int H_out,
    int W_out) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    int oc = (index / (H_out * W_out)) % C_out;
    output[index] += bias[oc];
}

// Forward function that tunes block sizes based on occupancy for optimal performance
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
    if(bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
    }

    // Input dimensions: [N, C_in, H_in, W_in]
    auto x_sizes = x.sizes();
    int N = x_sizes[0];
    int C_in = x_sizes[1];
    int H_in = x_sizes[2];
    int W_in = x_sizes[3];

    // Weight dimensions (assumed square kernel): [C_in, C_out, K, K]
    auto w_sizes = weight.sizes();
    int C_out = w_sizes[1];
    int K = w_sizes[2];

    // Compute output dimensions for transposed convolution
    int H_out = (H_in - 1) * stride - 2 * padding + K + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + K + output_padding;

    // Allocate output tensor
    auto output = torch::empty({N, C_out, H_out, W_out}, x.options());
    int total_output = N * C_out * H_out * W_out;

    // Determine optimal block size using CUDA occupancy API for the main kernel
    int optimal_block_size = 256; // default
    int min_grid_size = 0;
    cudaError_t occ_err = cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &optimal_block_size, conv_transpose2d_kernel_output, 0, 0);
    if (occ_err != cudaSuccess) {
        optimal_block_size = 256;
    }
    int grid_size = (total_output + optimal_block_size - 1) / optimal_block_size;

    // Launch the transposed convolution kernel
    conv_transpose2d_kernel_output<<<grid_size, optimal_block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, K,
        stride, padding,
        H_out, W_out
    );
    cudaDeviceSynchronize();

    // If bias is provided, add it using an optimized block configuration
    if (bias.has_value()) {
        int optimal_block_size_bias = optimal_block_size; // reuse same block size
        int grid_size_bias = (total_output + optimal_block_size_bias - 1) / optimal_block_size_bias;
        add_bias_kernel<<<grid_size_bias, optimal_block_size_bias>>>(
            output.data_ptr<float>(),
            bias.value().data_ptr<float>(),
            total_output, C_out, H_out, W_out
        );
        cudaDeviceSynchronize();
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA) - tuned block sizes");
}
