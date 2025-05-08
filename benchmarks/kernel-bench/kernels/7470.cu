#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tunable parameter for custom kernel
#define BLOCK_SIZE 512

// Device function: compute one element of the output
__device__ inline float compute_output_element(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int B, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int stride, int padding,
    int b, int oc, int h, int w) {

    float sum = 0.0f;
    int h_offset = h + padding;
    int w_offset = w + padding;
    
    // Pre-compute base indices for better memory access patterns
    int base_input = b * (C_in * H_in * W_in);
    int weight_oc_offset = oc * (K * K);
    
    // Reorder loops to improve memory coalescing
    // Process kernel window first to maximize spatial locality
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
            
            // Process all input channels for this spatial location
            for (int ic = 0; ic < C_in; ++ic) {
                int input_idx = base_input + ic * (H_in * W_in) + h_in * W_in + w_in;
                int weight_idx = ic * (C_out * K * K) + weight_oc_offset + kh * K + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    return sum;
}

// Custom tuned CUDA kernel for ConvTranspose2d
__global__ void conv_transpose2d_kernel_tuned(
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

    // Decode the output index into (b, oc, h, w)
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

// Custom forward function using the custom CUDA kernel
// This implementation supports only groups==1 and output_padding==0
torch::Tensor conv_transpose2d_forward_custom(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    // Ensure the custom kernel is used only with supported parameters
    TORCH_CHECK(groups == 1, "Custom kernel supports only groups==1");
    TORCH_CHECK(output_padding == 0, "Custom kernel supports only output_padding==0");

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
    int threads = BLOCK_SIZE;
    int blocks = (total_outputs + threads - 1) / threads;

    conv_transpose2d_kernel_tuned<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output_tensor.data_ptr<float>(),
        B, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K, stride, padding);

    return output_tensor;
}

// Combined forward function: choose the custom kernel or fallback to ATen's implementation
torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    // Check inputs
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
    }

    // Use custom kernel if parameters are supported, otherwise fallback
    if (groups == 1 && output_padding == 0) {
        return conv_transpose2d_forward_custom(input, weight, bias, stride, padding, output_padding, groups);
    } else {
        return at::conv_transpose2d(
            input,
            weight,
            bias,
            {stride, stride},
            {padding, padding},
            {output_padding, output_padding},
            groups);
    }
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Combined ConvTranspose2d forward (CUDA)");
}
