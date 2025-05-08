#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__device__ __forceinline__ void calculate_output_position(
    int output_idx, int W_out, int H_out, int C_out,
    int& w_out, int& h_out, int& c_out, int& n
) {
    w_out = output_idx % W_out;
    int tmp = output_idx / W_out;
    h_out = tmp % H_out;
    tmp = tmp / H_out;
    c_out = tmp % C_out;
    n = tmp / C_out;
}

__device__ __forceinline__ void calculate_group_info(
    int c_out, int C_out, int C_in, int groups,
    int& group, int& c_in_start, int& c_in_end
) {
    group = c_out / (C_out / groups);
    c_in_start = group * (C_in / groups);
    c_in_end = c_in_start + (C_in / groups);
}

__device__ __forceinline__ bool is_valid_position(
    int h_in, int w_in, int H_in, int W_in
) {
    return (h_in >= 0) && (h_in < H_in) && (w_in >= 0) && (w_in < W_in);
}

__device__ __forceinline__ float compute_conv_value(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int n, int c_in_start, int c_in_end, int C_in,
    int h_out, int w_out, int H_in, int W_in,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int C_in_per_group, int c_out
) {
    float value = 0.0f;
    
    const int h_in_base = h_out * stride_h - padding_h;
    const int w_in_base = w_out * stride_w - padding_w;

    #pragma unroll
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        const int input_channel_offset = ((n * C_in + c_in) * H_in) * W_in;
        const int weight_channel_offset = ((c_out * C_in_per_group + (c_in - c_in_start)) * K_h) * K_w;

        #pragma unroll
        for (int k_h = 0; k_h < K_h; ++k_h) {
            const int h_in = h_in_base + k_h * dilation_h;
            
            if (h_in >= 0 && h_in < H_in) {
                const int input_h_offset = input_channel_offset + h_in * W_in;
                const int weight_h_offset = weight_channel_offset + k_h * K_w;

                #pragma unroll
                for (int k_w = 0; k_w < K_w; ++k_w) {
                    const int w_in = w_in_base + k_w * dilation_w;
                    
                    if (w_in >= 0 && w_in < W_in) {
                        value += input[input_h_offset + w_in] * 
                                weight[weight_h_offset + k_w];
                    }
                }
            }
        }
    }
    
    return value;
}

__global__ void conv2d_cuda_kernel_strided(
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
    int stride = gridDim.x * blockDim.x;

    for (int output_idx = index; output_idx < N * C_out * H_out * W_out; output_idx += stride) {
        int w_out, h_out, c_out, n;
        calculate_output_position(output_idx, W_out, H_out, C_out, w_out, h_out, c_out, n);

        int group, c_in_start, c_in_end;
        calculate_group_info(c_out, C_out, C_in, groups, group, c_in_start, c_in_end);

        const int C_in_per_group = C_in / groups;
        
        float value = compute_conv_value(
            input, weight,
            n, c_in_start, c_in_end, C_in,
            h_out, w_out, H_in, W_in,
            K_h, K_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            C_in_per_group, c_out
        );

        if (bias != nullptr) {
            value += bias[c_out];
        }

        output[output_idx] = value;
    }
}

torch::Tensor conv2d_cuda_strided(
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

    const auto N = input.size(0);
    const auto C_in = input.size(1);
    const auto H_in = input.size(2);
    const auto W_in = input.size(3);
    const auto C_out = weight.size(0);
    const auto K_h = weight.size(2);
    const auto K_w = weight.size(3);

    const auto stride_h = stride[0];
    const auto stride_w = stride[1];
    const auto padding_h = padding[0];
    const auto padding_w = padding[1];
    const auto dilation_h = dilation[0];
    const auto dilation_w = dilation[1];

    const auto H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    const auto W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value().contiguous();
        bias_ptr = bias.data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    const int threads_per_block = 256;
    const int total_elements = N * C_out * H_out * W_out;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    conv2d_cuda_kernel_strided<<<num_blocks, threads_per_block>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda_strided, "Strided 2D convolution (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}