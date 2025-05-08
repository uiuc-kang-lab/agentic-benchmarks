#include <torch/extension.h>

// Templated kernel for bias addition with configurable block size
template <int BLOCK_SIZE>
__global__ void add_bias_kernel_opt(float* output, const float* bias, int total, int C_out, int H_out, int W_out) {
    int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
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

    // Perform the main conv_transpose2d computation using the built-in function
    auto output = at::conv_transpose2d(
        x,
        weight,
        bias,
        {stride, stride},                   // stride
        {padding, padding},                   // padding
        {output_padding, output_padding},     // output_padding
        groups
    );

    // If bias is provided, execute the bias addition kernel with an optimal block size
    if (bias.has_value()) {
        int N = x.size(0);
        int C_out = weight.size(1);
        int H_out = output.size(2);
        int W_out = output.size(3);
        int total_output = N * C_out * H_out * W_out;

        // Choose block size based on a simple heuristic
        int optimal_block_size;
        if (total_output <= 1024)
            optimal_block_size = 32;
        else if (total_output <= 4096)
            optimal_block_size = 64;
        else if (total_output <= 16384)
            optimal_block_size = 128;
        else if (total_output <= 65536)
            optimal_block_size = 256;
        else
            optimal_block_size = 512;

        int grid_size = (total_output + optimal_block_size - 1) / optimal_block_size;

        // Launch the templated kernel based on the selected block size
        switch(optimal_block_size) {
            case 32:
                add_bias_kernel_opt<32><<<grid_size, 32>>>(
                    output.data_ptr<float>(),
                    bias.value().data_ptr<float>(),
                    total_output, C_out, H_out, W_out
                );
                break;
            case 64:
                add_bias_kernel_opt<64><<<grid_size, 64>>>(
                    output.data_ptr<float>(),
                    bias.value().data_ptr<float>(),
                    total_output, C_out, H_out, W_out
                );
                break;
            case 128:
                add_bias_kernel_opt<128><<<grid_size, 128>>>(
                    output.data_ptr<float>(),
                    bias.value().data_ptr<float>(),
                    total_output, C_out, H_out, W_out
                );
                break;
            case 256:
                add_bias_kernel_opt<256><<<grid_size, 256>>>(
                    output.data_ptr<float>(),
                    bias.value().data_ptr<float>(),
                    total_output, C_out, H_out, W_out
                );
                break;
            case 512:
                add_bias_kernel_opt<512><<<grid_size, 512>>>(
                    output.data_ptr<float>(),
                    bias.value().data_ptr<float>(),
                    total_output, C_out, H_out, W_out
                );
                break;
            default:
                add_bias_kernel_opt<256><<<grid_size, 256>>>(
                    output.data_ptr<float>(),
                    bias.value().data_ptr<float>(),
                    total_output, C_out, H_out, W_out
                );
        }
        cudaDeviceSynchronize();
    }

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA) - block size experiment");
}
