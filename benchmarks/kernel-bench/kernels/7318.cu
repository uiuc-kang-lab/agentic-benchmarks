#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Constant memory for weights and bias
__constant__ float const_weight[8192];  // 32KB
__constant__ float const_bias[1024];    // 4KB

__global__ void conv2d_cuda_kernel_constant_mem(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    bool has_bias
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

    // Initialize accumulator with bias if present
    float value = has_bias ? const_bias[c_out] : 0.0f;

    // Calculate group information
    const int group = c_out / (C_out / groups);
    const int c_in_start = group * (C_in / groups);
    const int c_in_end = c_in_start + (C_in / groups);
    const int C_in_per_group = C_in / groups;

    // Pre-calculate base indices for input
    const int h_in_base = h_out * stride_h - padding_h;
    const int w_in_base = w_out * stride_w - padding_w;

    #pragma unroll 4
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
                                const_weight[weight_h_offset + k_w];
                    }
                }
            }
        }
    }

    output[index] = value;
}

torch::Tensor conv2d_cuda_constant_mem(
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

    // Check if weight size fits in constant memory
    TORCH_CHECK(weight.numel() <= 8192, "Weight tensor too large for constant memory");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().numel() <= 1024, "Bias tensor too large for constant memory");
    }

    // Copy weight to constant memory
    cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), 
                       weight.numel() * sizeof(float));

    bool has_bias = bias_opt.has_value();
    if (has_bias) {
        auto bias = bias_opt.value().contiguous();
        cudaMemcpyToSymbol(const_bias, bias.data_ptr<float>(), 
                          bias.numel() * sizeof(float));
    }

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
    float* output_ptr = output.data_ptr<float>();

    const int threads_per_block = 256;
    const int total_elements = N * C_out * H_out * W_out;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    conv2d_cuda_kernel_constant_mem<<<num_blocks, threads_per_block>>>(
        input_ptr, output_ptr,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups,
        has_bias
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda_constant_mem, "Constant memory optimized 2D convolution (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1},
          py::arg("groups") = 1);
}