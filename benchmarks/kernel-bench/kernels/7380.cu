#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Device function to calculate input indices
__device__ __forceinline__ bool calculate_input_idx(
    int h_out, int w_out,
    int k_h, int k_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int H_in, int W_in,
    int& h_in, int& w_in
) {
    h_in = h_out * stride_h - padding_h + k_h * dilation_h;
    w_in = w_out * stride_w - padding_w + k_w * dilation_w;
    return (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in);
}

// Device function to calculate output index
__device__ __forceinline__ int calculate_output_idx(
    int n, int c_out, int h_out, int w_out,
    int C_out, int H_out, int W_out
) {
    return ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
}

// Device function to calculate weight index
__device__ __forceinline__ int calculate_weight_idx(
    int c_out, int c_in, int c_in_start,
    int k_h, int k_w, int K_h, int K_w,
    int C_in, int groups
) {
    return (((c_out * (C_in / groups) + (c_in - c_in_start)) * K_h + k_h) * K_w) + k_w;
}

// Main convolution kernel
__global__ void conv2d_cuda_kernel(
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
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * C_out * H_out * W_out) return;

    // Calculate output position
    const int w_out = tid % W_out;
    int tmp = tid / W_out;
    const int h_out = tmp % H_out;
    tmp = tmp / H_out;
    const int c_out = tmp % C_out;
    const int n = tmp / C_out;

    // Initialize output value with bias if present
    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Calculate group information
    const int group = c_out / (C_out / groups);
    const int c_in_start = group * (C_in / groups);
    const int c_in_end = c_in_start + (C_in / groups);

    // Main convolution loop
    #pragma unroll 4
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        for (int k_h = 0; k_h < K_h; ++k_h) {
            for (int k_w = 0; k_w < K_w; ++k_w) {
                int h_in, w_in;
                if (calculate_input_idx(h_out, w_out, k_h, k_w,
                                     stride_h, stride_w,
                                     padding_h, padding_w,
                                     dilation_h, dilation_w,
                                     H_in, W_in,
                                     h_in, w_in)) {
                    const int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    const int weight_idx = calculate_weight_idx(c_out, c_in, c_in_start,
                                                             k_h, k_w, K_h, K_w,
                                                             C_in, groups);
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Write output
    const int output_idx = calculate_output_idx(n, c_out, h_out, w_out, C_out, H_out, W_out);
    output[output_idx] = value;
}

torch::Tensor conv2d_cuda(
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

    const int64_t N = input.size(0);
    const int64_t C_in = input.size(1);
    const int64_t H_in = input.size(2);
    const int64_t W_in = input.size(3);
    const int64_t C_out = weight.size(0);
    const int64_t K_h = weight.size(2);
    const int64_t K_w = weight.size(3);

    const int64_t H_out = (H_in + 2 * padding[0] - dilation[0] * (K_h - 1) - 1) / stride[0] + 1;
    const int64_t W_out = (W_in + 2 * padding[1] - dilation[1] * (K_w - 1) - 1) / stride[1] + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const int threads = 256;
    const int blocks = (N * C_out * H_out * W_out + threads - 1) / threads;

    conv2d_cuda_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_opt.has_value() ? bias_opt.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda, "Modular 2D convolution (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}