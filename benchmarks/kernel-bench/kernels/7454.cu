#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized transposed convolution kernel with memory coalescing
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int stride, int padding) {

    // Calculate the output element this thread block is responsible for
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;

    // Load weight for the current output channel into shared memory
    extern __shared__ float sh_weight[];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int ic = 0; ic < C_in; ++ic) {
            for (int r = 0; r < K*K; ++r) {
                sh_weight[ic * (K*K) + r] = weight[ic * (C_out * K * K) + oc * (K*K) + r];
            }
        }
    }
    __syncthreads();

    if (w < W_out && h < H_out) {
        for (int b = 0; b < B; ++b) {
            float sum = 0.0f;
            for (int ic = 0; ic < C_in; ++ic) {
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        int h_in = h + padding - kh;
                        int w_in = w + padding - kw;
                        if ((h_in % stride == 0) && (w_in % stride == 0)) {
                            h_in /= stride;
                            w_in /= stride;
                            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                int input_idx = b * (C_in * H_in * W_in) + ic * (H_in * W_in) + h_in * W_in + w_in;
                                int weight_idx = ic * (C_out * K * K) + oc * (K * K) + kh * K + kw;
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
            if (bias != nullptr) {
                sum += bias[oc];
            }
            int out_offset = b * (C_out * H_out * W_out) + oc * (H_out * W_out) + h * W_out + w;
            output[out_offset] = sum;
        }
    }
}

// Forward function that wraps the optimized CUDA kernel
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

    // This optimized kernel supports only groups=1 and output_padding==0 currently
    TORCH_CHECK(groups == 1, "Only groups==1 is supported in the optimized kernel");
    TORCH_CHECK(output_padding == 0, "Only output_padding==0 is supported in the optimized kernel");

    int B = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    // Assuming weight has shape (C_in, C_out, K, K) for square kernel
    int K = weight.size(2);
    int C_out = weight.size(1);
    int H_out = (H_in - 1) * stride - 2 * padding + K;
    int W_out = (W_in - 1) * stride - 2 * padding + K;

    auto output_tensor = torch::zeros({B, C_out, H_out, W_out}, input.options());

    dim3 threads(16, 16);
    dim3 blocks((W_out + threads.x - 1) / threads.x, (H_out + threads.y - 1) / threads.y, C_out);

    conv_transpose2d_kernel<<<blocks, threads>>>(
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
    m.def("forward", &conv_transpose2d_forward, "Optimized ConvTranspose2d forward (CUDA) with memory coalescing");
}