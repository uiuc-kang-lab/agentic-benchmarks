#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tunable block size
#define BLOCK_SIZE 512

// Modular device function to accumulate contributions from a single input channel
__device__ inline float accumulate_channel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int base_input_ic,
    int H_in, int W_in,
    int h_offset, int w_offset,
    int stride, int K) {

    float acc = 0.0f;
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
            int weight_idx = kh * K + kw;
            acc += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
        }
    }
    return acc;
}

// Modular device function to compute a single output element
__device__ inline float compute_output_element(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int b, int C_in, int H_in, int W_in,
    int oc, int h, int w,
    int K, int stride, int padding, int C_out) {

    float sum = 0.0f;
    int h_offset = h + padding;
    int w_offset = w + padding;

    // Loop over input channels and accumulate convolution results
    for (int ic = 0; ic < C_in; ++ic) {
        int base_input_ic = b * (C_in * H_in * W_in) + ic * (H_in * W_in);
        const float* weight_ptr = weight + ic * (C_out * K * K) + oc * (K * K);
        sum += accumulate_channel(input, weight_ptr, base_input_ic, H_in, W_in, h_offset, w_offset, stride, K);
    }
    return sum;
}

// CUDA kernel for transposed convolution using modular device functions
__global__ void conv_transpose2d_kernel_modular(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int stride, int padding) {

    int total_outputs = B * C_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) return;

    // Decode the linear index into 4D indices: (b, oc, h, w)
    int w = idx % W_out;
    idx /= W_out;
    int h = idx % H_out;
    idx /= H_out;
    int oc = idx % C_out;
    int b = idx / C_out;

    float out_val = compute_output_element(input, weight, b, C_in, H_in, W_in, oc, h, w, K, stride, padding, C_out);
    if (bias != nullptr) {
        out_val += bias[oc];
    }

    int out_offset = b * (C_out * H_out * W_in) + oc * (H_out * W_out) + h * W_out + w;
    // Note: Corrected the multiplication factor for channel stride
    out_offset = b * (C_out * H_out * W_out) + oc * (H_out * W_out) + h * W_out + w;

    output[out_offset] = out_val;
}

// Forward function wrapping the modular CUDA kernel
// This implementation supports only groups==1 and output_padding==0

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

    TORCH_CHECK(groups == 1, "Only groups==1 is supported in the modular kernel");
    TORCH_CHECK(output_padding == 0, "Only output_padding==0 is supported in the modular kernel");

    int B = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int K = weight.size(2);
    int C_out = weight.size(1);

    int H_out = (H_in - 1) * stride - 2 * padding + K;
    int W_out = (W_in - 1) * stride - 2 * padding + K;

    auto output_tensor = torch::zeros({B, C_out, H_out, W_out}, input.options());

    int total = B * C_out * H_out * W_out;
    int threads = BLOCK_SIZE;
    int blocks = (total + threads - 1) / threads;

    conv_transpose2d_kernel_modular<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output_tensor.data_ptr<float>(),
        B, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K, stride, padding);

    return output_tensor;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Modular ConvTranspose2d forward (CUDA) using device functions");
}
