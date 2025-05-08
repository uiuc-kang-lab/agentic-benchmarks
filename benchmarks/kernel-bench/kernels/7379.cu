#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

// Branchless 2D convolution kernel that minimizes warp divergence by using arithmetic
// boundary checks rather than conditional branches inside the inner loops. The kernel
// uses unsigned comparisons to compute validity masks, and the ternary operator (which
// modern compilers can convert to predicated instructions) to avoid out-of-bound memory
// accesses, ensuring uniform control flow within a warp.

__global__ void conv2d_cuda_kernel_branchless(
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
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (index >= total) return;

    // Decode the flat index into (n, c_out, h_out, w_out) coordinates
    int w_out = index % W_out;
    int tmp = index / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    // Initialize output value with bias (if provided) or zero
    float out_val = (bias != nullptr) ? bias[c_out] : 0.0f;

    int channels_per_group = C_in / groups;
    int group = c_out / (C_out / groups);
    int c_in_start = group * channels_per_group;

    // Loop over the input channels within the group and over the kernel height/width
    for (int c_in = c_in_start; c_in < c_in_start + channels_per_group; c_in++) {
        for (int kh = 0; kh < K_h; kh++) {
            int h_in = h_out * stride_h - padding_h + kh * dilation_h;
            // Branchless check: valid if 0 <= h_in < H_in
            int valid_h = ((unsigned)h_in < (unsigned)H_in);
            for (int kw = 0; kw < K_w; kw++) {
                int w_in = w_out * stride_w - padding_w + kw * dilation_w;
                // Branchless check: valid if 0 <= w_in < W_in
                int valid_w = ((unsigned)w_in < (unsigned)W_in);
                int valid = valid_h & valid_w;

                // Use the ternary operator to avoid out-of-bound memory accesses.
                // If the coordinate is invalid, the loaded value is 0.0f.
                int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                int weight_idx = (((c_out * channels_per_group) + (c_in - c_in_start)) * K_h + kh) * K_w + kw;
                float in_val = valid ? input[input_idx] : 0.0f;
                float wt_val = valid ? weight[weight_idx] : 0.0f;
                out_val += in_val * wt_val;
            }
        }
    }

    int out_index = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[out_index] = out_val;
}

// C++ interface to the PyTorch module
torch::Tensor conv2d_cuda_branchless(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    input = input.contiguous();
    weight = weight.contiguous();

    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");

    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "Bias tensor must be on CUDA if provided");
    }

    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(0);
    int K_h = weight.size(2);
    int K_w = weight.size(3);

    int stride_h = stride[0];
    int stride_w = stride[1];
    int padding_h = padding[0];
    int padding_w = padding[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];

    int H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    int W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    int total_threads = N * C_out * H_out * W_out;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    conv2d_cuda_kernel_branchless<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_opt.has_value() ? bias_opt.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
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
        printf("Error in conv2d_cuda_kernel_branchless: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda_branchless, "Branchless 2D convolution kernel (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1},
          py::arg("groups") = 1);
}
