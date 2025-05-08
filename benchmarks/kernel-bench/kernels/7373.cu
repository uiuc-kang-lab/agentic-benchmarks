#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

// Define block size macro for tuning. This can be overridden via compile flags (e.g., -DBLOCK_SIZE=128)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// Optimized CUDA kernel for 2D convolution using tuned block size and memory coalescing
__global__ void conv2d_cuda_kernel_coalesced(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    int n = blockIdx.x;
    int c_out = blockIdx.y;
    int h_out = blockIdx.z * blockDim.y + threadIdx.y;
    int w_out = threadIdx.x;

    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    int group = c_out / (C_out / groups);
    int c_in_start = group * (C_in / groups);
    int c_in_end = c_in_start + (C_in / groups);

    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        for (int k_h = 0; k_h < K_h; ++k_h) {
            int h_in = h_out * stride_h - padding_h + k_h * dilation_h;
            if (h_in >= 0 && h_in < H_in) {
                for (int k_w = 0; k_w < K_w; ++k_w) {
                    int w_in = w_out * stride_w - padding_w + k_w * dilation_w;
                    if (w_in >= 0 && w_in < W_in) {
                        int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                        int weight_idx = (((c_out * (C_in / groups) + (c_in - c_in_start)) * K_h + k_h) * K_w) + k_w;
                        value += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[output_idx] = value;
}

// C++ interface to the PyTorch module
torch::Tensor conv2d_cuda_coalesced(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    // Ensure inputs are contiguous and on CUDA
    input = input.contiguous();
    weight = weight.contiguous();

    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");

    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "Bias tensor must be on CUDA if provided");
    }

    // Input dimensions
    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);

    // Weight dimensions
    int64_t C_out = weight.size(0);
    int64_t K_h = weight.size(2);
    int64_t K_w = weight.size(3);

    // Stride, padding, and dilation settings
    int64_t stride_h = stride[0];
    int64_t stride_w = stride[1];
    int64_t padding_h = padding[0];
    int64_t padding_w = padding[1];
    int64_t dilation_h = dilation[0];
    int64_t dilation_w = dilation[1];

    // Calculate output dimensions
    int64_t H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    int64_t W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    // Allocate output tensor
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    torch::Tensor bias;
    if (bias_opt.has_value()) {
        bias = bias_opt.value().contiguous();
        bias_ptr = bias.data_ptr<float>();
    }

    float* output_ptr = output.data_ptr<float>();

    // Launch the optimized kernel with memory coalescing
    dim3 threads_per_block(W_out, H_out);
    dim3 num_blocks(N, C_out);

    conv2d_cuda_kernel_coalesced<<<num_blocks, threads_per_block>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in conv2d_cuda_kernel_coalesced: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda_coalesced, "Optimized 2D convolution with memory coalescing (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}