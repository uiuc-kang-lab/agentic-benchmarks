#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__forceinline__ __device__ float4 load_input4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__global__ void conv2d_cuda_kernel_aligned(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out,
    const int K_h, const int K_w,
    const int stride_h, const int stride_w,
    const int padding_h, const int padding_w,
    const int dilation_h, const int dilation_w,
    const int groups
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = N * C_out * H_out * W_out;
    if (index >= total_elements) return;

    const int w_out = index % W_out;
    int temp = index / W_out;
    const int h_out = temp % H_out;
    temp /= H_out;
    const int c_out = temp % C_out;
    const int n = temp / C_out;

    float value = (bias != nullptr) ? __ldg(&bias[c_out]) : 0.0f;

    const int group = c_out / (C_out / groups);
    const int c_in_start = group * (C_in / groups);
    const int c_in_end = c_in_start + (C_in / groups);
    const int C_in_per_group = C_in / groups;

    const int h_in_base = h_out * stride_h - padding_h;
    const int w_in_base = w_out * stride_w - padding_w;

    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        const int input_channel_offset = ((n * C_in + c_in) * H_in) * W_in;
        const int weight_channel_offset = ((c_out * C_in_per_group + (c_in - c_in_start)) * K_h) * K_w;

        #pragma unroll
        for (int k_h = 0; k_h < K_h; ++k_h) {
            const int h_in = h_in_base + k_h * dilation_h;
            
            if (h_in >= 0 && h_in < H_in) {
                const int input_h_offset = input_channel_offset + h_in * W_in;
                const int weight_h_offset = weight_channel_offset + k_h * K_w;

                int k_w = 0;
                while (k_w + 3 < K_w) {
                    const int w_in = w_in_base + k_w * dilation_w;
                    if (w_in >= 0 && w_in + 3 * dilation_w < W_in) {
                        const float4 weight4 = load_input4(&weight[weight_h_offset + k_w]);
                        value += __ldg(&input[input_h_offset + w_in]) * weight4.x;
                        value += __ldg(&input[input_h_offset + w_in + dilation_w]) * weight4.y;
                        value += __ldg(&input[input_h_offset + w_in + 2 * dilation_w]) * weight4.z;
                        value += __ldg(&input[input_h_offset + w_in + 3 * dilation_w]) * weight4.w;
                    }
                    k_w += 4;
                }

                for (; k_w < K_w; ++k_w) {
                    const int w_in = w_in_base + k_w * dilation_w;
                    if (w_in >= 0 && w_in < W_in) {
                        value += __ldg(&input[input_h_offset + w_in]) * 
                                __ldg(&weight[weight_h_offset + k_w]);
                    }
                }
            }
        }
    }

    output[index] = value;
}

torch::Tensor conv2d_cuda_aligned(
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

    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int C_out = weight.size(0);
    const int K_h = weight.size(2);
    const int K_w = weight.size(3);

    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int padding_h = padding[0];
    const int padding_w = padding[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];

    const int H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    const int W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

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

    conv2d_cuda_kernel_aligned<<<num_blocks, threads_per_block>>>(
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
    m.def("forward", &conv2d_cuda_aligned, "Aligned memory access 2D convolution (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1},
          py::arg("groups") = 1);
}