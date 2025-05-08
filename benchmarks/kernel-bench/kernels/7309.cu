#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv2d_cuda_kernel_adaptive(
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
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = N * C_out * H_out * W_out;
    if (index >= total_elements) return;

    int w_out = index % W_out;
    int temp = index / W_out;
    int h_out = temp % H_out;
    temp /= H_out;
    int c_out = temp % C_out;
    int n = temp / C_out;

    float value = (bias != nullptr) ? bias[c_out] : 0.0f;
    int group = c_out / (C_out / groups);
    int c_in_start = group * (C_in / groups);
    int c_in_end = c_in_start + (C_in / groups);
    int C_in_per_group = C_in / groups;

    int h_in_origin = h_out * stride_h - padding_h;
    int w_in_origin = w_out * stride_w - padding_w;

    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        int input_offset = ((n * C_in + c_in) * H_in) * W_in;
        int weight_offset = ((c_out * C_in_per_group + (c_in - c_in_start)) * K_h) * K_w;

        for (int k_h = 0; k_h < K_h; ++k_h) {
            int h_in = h_in_origin + k_h * dilation_h;
            if (h_in >= 0 && h_in < H_in) {
                int input_line_offset = input_offset + h_in * W_in;
                int weight_line_offset = weight_offset + k_h * K_w;

                for (int k_w = 0; k_w < K_w; ++k_w) {
                    int w_in = w_in_origin + k_w * dilation_w;
                    if (w_in >= 0 && w_in < W_in) {
                        value += input[input_line_offset + w_in] * weight[weight_line_offset + k_w];
                    }
                }
            }
        }
    }

    output[index] = value;
}

torch::Tensor conv2d_cuda_adaptive(
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

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value().contiguous();
        bias_ptr = bias.data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    int total_elements = N * C_out * H_out * W_out;
    int blocks = (total_elements + 255) / 256;
    conv2d_cuda_kernel_adaptive<<<blocks, 256>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
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
        printf("Error in conv2d_cuda_kernel_adaptive: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda_adaptive, "Adaptive partitioned 2D convolution (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1},
          py::arg("groups") = 1);
}
