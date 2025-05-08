#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes one output element of the transposed convolution using manually unrolled loops for the kernel dimensions
// The inner loops over the kernel spatial dimensions (ki and kj) are unrolled to reduce loop overhead and improve performance.

__global__ void conv_transpose2d_kernel_unrolled(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
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

    // Decode the linear index into (n, oc, h, w)
    int w = index % W_out;
    int tmp = index / W_out;
    int h = tmp % H_out;
    tmp /= H_out;
    int oc = tmp % C_out;
    int n = tmp / C_out;

    float sum = 0.f;
    
    // Loop over the input channels
    for (int c = 0; c < C_in; c++) {
        // Unroll the kernel height dimension
        #pragma unroll
        for (int ki = 0; ki < K; ki++) {
            // Unroll the kernel width dimension
            #pragma unroll
            for (int kj = 0; kj < K; kj++) {
                // Compute the corresponding input coordinates
                int in_h = h + padding - ki;
                int in_w = w + padding - kj;
                
                // Only proceed if the coordinates align with the stride
                if (in_h % stride != 0 || in_w % stride != 0)
                    continue;
                in_h /= stride;
                in_w /= stride;
                
                // Check bounds for input
                if (in_h < 0 || in_h >= H_in || in_w < 0 || in_w >= W_in)
                    continue;
                
                int input_idx = n * (C_in * H_in * W_in) + c * (H_in * W_in) + in_h * W_in + in_w;
                int weight_idx = c * (C_out * K * K) + oc * (K * K) + ki * K + kj;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    output[index] = sum;
}

// Kernel to add bias to each output channel
__global__ void add_bias_kernel_unrolled(
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

// Forward function for transposed convolution with manual loop unrolling in the critical loops
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
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
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
    int K = w_sizes[2];  // Assumes square kernel

    // Compute output dimensions for transposed convolution:
    // output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
    int H_out = (H_in - 1) * stride - 2 * padding + K + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + K + output_padding;

    auto output = torch::empty({N, C_out, H_out, W_out}, x.options());
    int total_output = N * C_out * H_out * W_out;
    int block_size = 256;
    int grid_size = (total_output + block_size - 1) / block_size;

    conv_transpose2d_kernel_unrolled<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, K, stride, padding,
        H_out, W_out
    );
    cudaDeviceSynchronize();

    if (bias.has_value()) {
        auto bias_tensor = bias.value();
        grid_size = (total_output + block_size - 1) / block_size;
        add_bias_kernel_unrolled<<<grid_size, block_size>>>(
            output.data_ptr<float>(),
            bias_tensor.data_ptr<float>(),
            total_output, C_out, H_out, W_out
        );
        cudaDeviceSynchronize();
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA) with loop unrolling");
}
