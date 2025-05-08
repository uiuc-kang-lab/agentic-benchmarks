#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the block size for tuning; experiment with values: 32, 64, 128, 256, 512
// For the current hardware (NVIDIA H100 80GB HBM3 with CUDA 12.2), 512 is chosen based on profiling.
#define BLOCK_SIZE 512

// Device function to compute a single output element of the transposed convolution
__device__ float compute_output_element(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int B, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int stride, int padding,
    int b, int oc, int h, int w) {

    float sum = 0.0f;
    int h_offset = h + padding;
    int w_offset = w + padding;
    int base_input = b * (C_in * H_in * W_in);
    for (int ic = 0; ic < C_in; ++ic) {
        int base_input_ic = base_input + ic * (H_in * W_in);
        int base_weight = ic * (C_out * K * K) + oc * (K * K);
        for (int kh = 0; kh < K; ++kh) {
            int h_in_candidate = h_offset - kh;
            if (h_in_candidate % stride != 0) continue;
            int h_in = h_in_candidate / stride;
            if (h_in < 0 || h_in >= H_in) continue;
            for (int kw = 0; kw < K; ++kw) {
                int w_in_candidate = w_offset - kw;
                if (w_in_candidate % stride != 0) continue;
                int w_in = w_in_candidate / stride;
                if (w_in < 0 || w_in >= W_in) continue;
                int input_idx = base_input_ic + h_in * W_in + w_in;
                int weight_idx = base_weight + kh * K + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    return sum;
}

// Optimized kernel for transposed convolution using a tuned block size
__global__ void conv_transpose2d_kernel_optimized(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int stride, int padding) {

    int total_outputs = B * C_out * H_out * W_out;
    int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_index >= total_outputs) return;

    int w = out_index % W_out;
    out_index /= W_out;
    int h = out_index % H_out;
    out_index /= H_out;
    int oc = out_index % C_out;
    int b = out_index / C_out;

    float sum = compute_output_element(input, weight,
                                         B, C_in, H_in, W_in,
                                         C_out, H_out, W_out,
                                         K, stride, padding,
                                         b, oc, h, w);

    if (bias != nullptr) {
        sum += bias[oc];
    }

    int out_offset = b * (C_out * H_out * W_out) + oc * (H_out * W_out) + h * W_out + w;
    output[out_offset] = sum;
}

// Forward function wrapping the optimized CUDA kernel
torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
    }
    TORCH_CHECK(groups == 1, "Only groups==1 is supported in the optimized kernel");
    TORCH_CHECK(output_padding == 0, "Only output_padding==0 is supported in the optimized kernel");

    int B = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int K = weight.size(2);
    int C_out = weight.size(1);
    int H_out = (H_in - 1) * stride - 2 * padding + K;
    int W_out = (W_in - 1) * stride - 2 * padding + K;

    auto output_tensor = torch::zeros({B, C_out, H_out, W_out}, input.options());

    int total_outputs = B * C_out * H_out * W_out;
    int threads = BLOCK_SIZE;  // Tuning parameter; change to 32, 64, 128, 256, or 512 as needed
    int blocks = (total_outputs + threads - 1) / threads;

    conv_transpose2d_kernel_optimized<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output_tensor.data_ptr<float>(),
        B, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K, stride, padding);

    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Optimized ConvTranspose2d forward (CUDA)");
}
