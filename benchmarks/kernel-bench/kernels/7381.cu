#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define constant memory for weights and bias
__constant__ float const_weight[1024];
__constant__ float const_bias[256];

__global__ void conv2d_cuda_kernel_constant(
    const float* __restrict__ input,
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
    if (index >= N * C_out * H_out * W_out) return;

    int w_out = index % W_out;
    int tmp = index / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    float value = const_bias[c_out];

    int group = c_out / (C_out / groups);
    int c_in_start = group * (C_in / groups);

    for (int c_in = c_in_start; c_in < c_in_start + (C_in / groups); ++c_in) {
        for (int k_h = 0; k_h < K_h; ++k_h) {
            for (int k_w = 0; k_w < K_w; ++k_w) {
                int h_in = h_out * stride_h - padding_h + k_h * dilation_h;
                int w_in = w_out * stride_w - padding_w + k_w * dilation_w;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    int weight_idx = (((c_out * (C_in / groups) + (c_in - c_in_start)) * K_h + k_h) * K_w) + k_w;
                    value += input[input_idx] * const_weight[weight_idx];
                }
            }
        }
    }

    int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[output_idx] = value;
}

// C++ interface to the PyTorch module
torch::Tensor conv2d_cuda_constant(
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

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);

    int64_t C_out = weight.size(0);
    int64_t K_h = weight.size(2);
    int64_t K_w = weight.size(3);

    int64_t stride_h = stride[0];
    int64_t stride_w = stride[1];
    int64_t padding_h = padding[0];
    int64_t padding_w = padding[1];
    int64_t dilation_h = dilation[0];
    int64_t dilation_w = dilation[1];

    int64_t H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    int64_t W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Copy weights and bias to constant memory
    cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), weight.numel() * sizeof(float));
    if (bias_opt.has_value()) {
        cudaMemcpyToSymbol(const_bias, bias_opt.value().data_ptr<float>(), bias_opt.value().numel() * sizeof(float));
    } else {
        {
        std::vector<float> zeros(C_out, 0);
        cudaMemcpyToSymbol(const_bias, zeros.data(), C_out * sizeof(float));
    }
    }

    int total_threads = N * C_out * H_out * W_out;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    conv2d_cuda_kernel_constant<<<blocks, threads_per_block>>>(
        input_ptr,
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
        printf("Error in conv2d_cuda_kernel_constant: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda_constant, "Constant memory 2D convolution (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}
