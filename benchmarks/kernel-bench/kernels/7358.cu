#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

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
    extern __shared__ float shared_weight[];
    
    // Calculate thread indices
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int total_threads = N * C_out * H_out * W_out;
    
    if (idx >= total_threads) return;

    // Pre-compute output indices
    const int w_out = idx % W_out;
    int tmp = idx / W_out;
    const int h_out = tmp % H_out;
    tmp = tmp / H_out;
    const int c_out = tmp % C_out;
    const int n = tmp / C_out;

    // Load bias value
    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Calculate group-related constants
    const int channels_per_group = C_in / groups;
    const int group = c_out / (C_out / groups);
    const int c_in_start = group * channels_per_group;
    const int c_in_end = c_in_start + channels_per_group;

    // Pre-compute base indices for input and weight
    const int input_batch_offset = n * C_in * H_in * W_in;
    const int weight_out_offset = c_out * channels_per_group * K_h * K_w;

    // Pre-compute output position in input space
    const int h_out_pos = h_out * stride_h - padding_h;
    const int w_out_pos = w_out * stride_w - padding_w;

    // Process convolution
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        const int input_channel_offset = input_batch_offset + c_in * H_in * W_in;
        const int weight_in_offset = weight_out_offset + (c_in - c_in_start) * K_h * K_w;

        #pragma unroll 4
        for (int k_h = 0; k_h < K_h; ++k_h) {
            const int h_in = h_out_pos + k_h * dilation_h;
            if (h_in >= 0 && h_in < H_in) {
                #pragma unroll 4
                for (int k_w = 0; k_w < K_w; ++k_w) {
                    const int w_in = w_out_pos + k_w * dilation_w;
                    if (w_in >= 0 && w_in < W_in) {
                        const int input_idx = input_channel_offset + h_in * W_in + w_in;
                        const int weight_idx = weight_in_offset + k_h * K_w + k_w;
                        value += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Write output
    const int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
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

    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "Bias tensor must be on CUDA if provided");
    }

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
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;

    if (bias_opt.has_value()) {
        auto bias = bias_opt.value().contiguous();
        bias_ptr = bias.data_ptr<float>();
    }

    float* output_ptr = output.data_ptr<float>();

    int total_threads = N * C_out * H_out * W_out;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Launch kernel in the stream
    conv2d_cuda_kernel<<<blocks, threads_per_block, 0, stream>>>(
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

    // Destroy the stream
    cudaStreamDestroy(stream);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda, "Custom 2D convolution (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}
