#include <torch/extension.h>

// Optimized kernel that minimizes warp divergence by ensuring uniform control flow
__global__ void add_bias_kernel_optimized(
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

// Forward function definition
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    // Ensure inputs are on CUDA and contiguous
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
    }

    // Use the built-in conv_transpose2d function for the main computation
    auto output = at::conv_transpose2d(
        x,
        weight,
        bias,
        {stride, stride},                   // stride
        {padding, padding},                 // padding
        {output_padding, output_padding},   // output_padding
        groups
    );

    // If bias is provided, add it using a separate kernel
    if (bias.has_value()) {
        int N = x.size(0);
        int C_out = weight.size(1);
        int H_out = output.size(2);
        int W_out = output.size(3);
        int total_output = N * C_out * H_out * W_out;
        int block_size = 256;
        int grid_size = (total_output + block_size - 1) / block_size;

        add_bias_kernel_optimized<<<grid_size, block_size>>>(
            output.data_ptr<float>(),
            bias.value().data_ptr<float>(),
            total_output, C_out, H_out, W_out
        );
        cudaDeviceSynchronize();
    }

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA) - warp optimized");
}
