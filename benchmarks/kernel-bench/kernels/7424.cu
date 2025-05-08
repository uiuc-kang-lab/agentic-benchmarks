#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel for ConvTranspose2d without bias, with inner loops unrolled using #pragma unroll
__global__ void conv_transpose2d_kernel_unrolled_no_bias(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K,
    int stride, int padding,
    int H_out, int W_out) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (index >= total) return;

    // Decode linear index into (n, oc, out_h, out_w)
    int out_w = index % W_out;
    int tmp = index / W_out;
    int out_h = tmp % H_out;
    tmp /= H_out;
    int oc = tmp % C_out;
    int n = tmp / C_out;

    float sum = 0.0f;
    
    // Loop over input channels
    for (int c = 0; c < C_in; ++c) {
        #pragma unroll
        for (int k_i = 0; k_i < K; ++k_i) {
            #pragma unroll
            for (int k_j = 0; k_j < K; ++k_j) {
                int in_h = out_h + padding - k_i;
                int in_w = out_w + padding - k_j;
                if ((in_h % stride != 0) || (in_w % stride != 0)) continue;
                in_h /= stride;
                in_w /= stride;
                if (in_h < 0 || in_h >= H_in || in_w < 0 || in_w >= W_in) continue;
                int input_idx = n * (C_in * H_in * W_in) + c * (H_in * W_in) + in_h * W_in + in_w;
                int weight_idx = c * (C_out * K * K) + oc * (K * K) + k_i * K + k_j;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    output[index] = sum;
}

// Kernel for ConvTranspose2d with bias fused, with inner loops unrolled
__global__ void conv_transpose2d_kernel_unrolled_bias(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const float* __restrict__ bias,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K,
    int stride, int padding,
    int H_out, int W_out) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (index >= total) return;

    // Decode linear index into (n, oc, out_h, out_w)
    int out_w = index % W_out;
    int tmp = index / W_out;
    int out_h = tmp % H_out;
    tmp /= H_out;
    int oc = tmp % C_out;
    int n = tmp / C_out;

    float sum = 0.0f;
    
    // Loop over input channels
    for (int c = 0; c < C_in; ++c) {
        #pragma unroll
        for (int k_i = 0; k_i < K; ++k_i) {
            #pragma unroll
            for (int k_j = 0; k_j < K; ++k_j) {
                int in_h = out_h + padding - k_i;
                int in_w = out_w + padding - k_j;
                if ((in_h % stride != 0) || (in_w % stride != 0)) continue;
                in_h /= stride;
                in_w /= stride;
                if (in_h < 0 || in_h >= H_in || in_w < 0 || in_w >= W_in) continue;
                int input_idx = n * (C_in * H_in * W_in) + c * (H_in * W_in) + in_h * W_in + in_w;
                int weight_idx = c * (C_out * K * K) + oc * (K * K) + k_i * K + k_j;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    output[index] = sum + bias[oc];
}

// ConvTranspose2d forward function with loop unrolling for inner loops
// Supports square input and square kernel, groups = 1

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
    if (bias.has_value()) {
        auto bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias_tensor.is_contiguous(), "Bias tensor must be contiguous");
    }

    // Input tensor shape: [N, C_in, H_in, W_in]
    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    // Weight tensor shape (assumed): [C_in, C_out, K, K]
    int C_out = weight.size(1);
    int K = weight.size(2);  // square kernel

    // Compute output dimensions:
    int H_out = (H_in - 1) * stride - 2 * padding + K + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + K + output_padding;

    auto output = torch::empty({N, C_out, H_out, W_out}, x.options());

    int total_elements = N * C_out * H_out * W_out;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    if (bias.has_value()) {
        conv_transpose2d_kernel_unrolled_bias<<<grid_size, block_size>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            bias.value().data_ptr<float>(),
            N, C_in, H_in, W_in,
            C_out, K,
            stride, padding,
            H_out, W_out
        );
    } else {
        conv_transpose2d_kernel_unrolled_no_bias<<<grid_size, block_size>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C_in, H_in, W_in,
            C_out, K,
            stride, padding,
            H_out, W_out
        );
    }

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward with loop unrolling (CUDA)");
}
