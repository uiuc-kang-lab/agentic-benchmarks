#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses stride loops to cover all output elements when the total workload exceeds the available threads.
// Each thread processes multiple output indices by incrementing the index by the total thread count. Correct boundary
// handling is ensured by computing each output's coordinates and checking validity for the corresponding input index.

__global__ void conv_transpose2d_kernel_stride(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const float* __restrict__ bias,  // Can be nullptr if no bias
    int N,
    int C_in, int H_in, int W_in,
    int C_out,
    int K,
    int stride,
    int padding,
    int H_out, int W_out) {

    int total = N * C_out * H_out * W_out;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_total = blockDim.x * gridDim.x;

    // Use a stride loop: each thread processes multiple output indices.
    for (int out_idx = tid; out_idx < total; out_idx += stride_total) {
        // Decode linear index into (n, oc, h, w)
        int w_out = out_idx % W_out;
        int tmp = out_idx / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int oc = tmp % C_out;
        int n = tmp / C_out;

        float sum = 0.0f;
        
        // Loop over input channels and kernel window
        for (int c = 0; c < C_in; ++c) {
            for (int ki = 0; ki < K; ++ki) {
                for (int kj = 0; kj < K; ++kj) {
                    // Compute the corresponding input coordinate
                    int in_h = h_out + padding - ki;
                    int in_w = w_out + padding - kj;
                    // Check if the coordinate aligns with the stride
                    if ((in_h % stride) != 0 || (in_w % stride) != 0) continue;
                    in_h /= stride;
                    in_w /= stride;
                    // Verify boundaries
                    if (in_h < 0 || in_h >= H_in || in_w < 0 || in_w >= W_in) continue;

                    int input_index = n * (C_in * H_in * W_in) + c * (H_in * W_in) + in_h * W_in + in_w;
                    int weight_index = c * (C_out * K * K) + oc * (K * K) + ki * K + kj;
                    sum += input[input_index] * weight[weight_index];
                }
            }
        }
        // Fuse bias addition if provided
        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[out_idx] = sum;
    }
}

// Forward function that sets up the CUDA kernel launch, using stride loops for workload partitioning.
// This function computes the transposed convolution operation with optional bias fused in the main kernel.

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias_opt.value().is_contiguous(), "Bias tensor must be contiguous");
    }

    // Input tensor shape: [N, C_in, H_in, W_in]
    auto x_sizes = x.sizes();
    int N = x_sizes[0];
    int C_in = x_sizes[1];
    int H_in = x_sizes[2];
    int W_in = x_sizes[3];

    // Weight tensor shape: [C_in, C_out, K, K] (square kernel assumed)
    auto w_sizes = weight.sizes();
    int C_out = w_sizes[1];
    int K = w_sizes[2];

    // Compute output dimensions for transposed convolution
    int H_out = (H_in - 1) * stride - 2 * padding + K + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + K + output_padding;

    auto output = torch::empty({N, C_out, H_out, W_out}, x.options());

    int total = N * C_out * H_out * W_out;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    conv_transpose2d_kernel_stride<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        bias_ptr,
        N, C_in, H_in, W_in,
        C_out, K,
        stride, padding,
        H_out, W_out
    );

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Transposed Conv2d forward (CUDA with stride loops)");
}
