#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

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
    // Reorganize indexing for coalesced memory access
    // Make threads within a warp access consecutive memory locations
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate position using width-first ordering for coalesced access
    int w_out = tid % W_out;
    int h_out = (tid / W_out) % H_out;
    int c_out = (tid / (W_out * H_out)) % C_out;
    int n = tid / (W_out * H_out * C_out);

    if (n >= N) return;

    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    int group = c_out / (C_out / groups);
    int c_in_start = group * (C_in / groups);
    int c_in_end = c_in_start + (C_in / groups);
    
    // Pre-calculate base indices for better memory access patterns
    int output_base = n * C_out * H_out * W_out + c_out * H_out * W_out;
    int input_base = n * C_in * H_in * W_in;
    int weight_base = c_out * (C_in / groups) * K_h * K_w;

    // Process input channels in chunks for better cache utilization
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        int input_channel_offset = input_base + c_in * H_in * W_in;
        int weight_channel_offset = weight_base + (c_in - c_in_start) * K_h * K_w;

        for (int k_h = 0; k_h < K_h; ++k_h) {
            int h_in = h_out * stride_h - padding_h + k_h * dilation_h;
            if (h_in >= 0 && h_in < H_in) {
                int input_h_offset = input_channel_offset + h_in * W_in;
                int weight_h_offset = weight_channel_offset + k_h * K_w;

                #pragma unroll
                for (int k_w = 0; k_w < K_w; ++k_w) {
                    int w_in = w_out * stride_w - padding_w + k_w * dilation_w;
                    if (w_in >= 0 && w_in < W_in) {
                        value += input[input_h_offset + w_in] *
                                weight[weight_h_offset + k_w];
                    }
                }
            }
        }
    }

    // Write output with coalesced access pattern
    if (tid < N * C_out * H_out * W_out) {
        output[output_base + h_out * W_out + w_out] = value;
    }
}

torch::Tensor conv2d_cuda_coalesced(
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

    auto dims = input.sizes();
    auto N = dims[0];
    auto C_in = dims[1];
    auto H_in = dims[2];
    auto W_in = dims[3];

    auto weight_dims = weight.sizes();
    auto C_out = weight_dims[0];
    auto K_h = weight_dims[2];
    auto K_w = weight_dims[3];

    auto stride_h = stride[0];
    auto stride_w = stride[1];
    auto padding_h = padding[0];
    auto padding_w = padding[1];
    auto dilation_h = dilation[0];
    auto dilation_w = dilation[1];

    auto H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    auto W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;

    if (bias_opt.has_value()) {
        auto bias = bias_opt.value().contiguous();
        bias_ptr = bias.data_ptr<float>();
    }

    float* output_ptr = output.data_ptr<float>();

    // Adjust block size to optimize for memory coalescing
    int threads_per_block = 256;
    int total_elements = N * C_out * H_out * W_out;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

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

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda_coalesced, "Coalesced memory access 2D convolution (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}