#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fused CUDA kernel that computes the transposed convolution and fuses bias addition
__global__ void conv_transpose2d_kernel_fused(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const float* __restrict__ bias,  // can be nullptr if no bias
    int N,
    int C_in, int H_in, int W_in,
    int C_out,
    int K,
    int stride,
    int padding,
    int H_out, int W_out) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (index >= total) return;

    // Decode output index into (n, oc, h, w)
    int w_out = index % W_out;
    int tmp = index / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int oc = tmp % C_out;
    int n = tmp / C_out;

    float sum = 0.f;
    // Loop over input channels and kernel window
    for (int c = 0; c < C_in; ++c) {
        for (int k_i = 0; k_i < K; ++k_i) {
            for (int k_j = 0; k_j < K; ++k_j) {
                // Compute corresponding input coordinate according to the reverse mapping
                int in_h = h_out + padding - k_i;
                int in_w = w_out + padding - k_j;
                // Only compute if the coordinates align with the stride
                if (in_h % stride != 0 || in_w % stride != 0) {
                    continue;
                }
                in_h /= stride;
                in_w /= stride;
                if (in_h < 0 || in_h >= H_in || in_w < 0 || in_w >= W_in) {
                    continue;
                }
                int input_idx = n * (C_in * H_in * W_in) + c * (H_in * W_in) + in_h * W_in + in_w;
                int weight_idx = c * (C_out * K * K) + oc * (K * K) + k_i * K + k_j;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    // Fuse bias addition if a valid bias pointer is provided
    if (bias != nullptr) {
        sum += bias[oc];
    }
    output[index] = sum;
}

// Transposed convolution forward function with fused bias addition
// This implementation supports groups = 1 only

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
    TORCH_CHECK(groups == 1, "Fused kernel currently supports groups = 1 only.");

    if (bias.has_value()) {
        auto bias_val = bias.value();
        TORCH_CHECK(bias_val.is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias_val.is_contiguous(), "Bias tensor must be contiguous");
    }

    // x: [N, C_in, H_in, W_in]
    auto x_sizes = x.sizes();
    int N = x_sizes[0];
    int C_in = x_sizes[1];
    int H_in = x_sizes[2];
    int W_in = x_sizes[3];

    // weight: [C_in, C_out, K, K]
    auto w_sizes = weight.sizes();
    int C_out = w_sizes[1];
    int K = w_sizes[2];

    // Compute output dimensions for transposed convolution
    int H_out = (H_in - 1) * stride - 2 * padding + K + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + K + output_padding;

    auto output = torch::empty({N, C_out, H_out, W_out}, x.options());
    int total_output = N * C_out * H_out * W_out;
    int block_size = 256;
    int grid_size = (total_output + block_size - 1) / block_size;

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    conv_transpose2d_kernel_fused<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        bias_ptr,
        N, C_in, H_in, W_in,
        C_out, K, stride, padding, H_out, W_out
    );

    // Optionally, one might remove explicit sync if the caller handles it
    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Fused ConvTranspose2d forward (CUDA)");
}
