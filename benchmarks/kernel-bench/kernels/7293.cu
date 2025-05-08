#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Combined kernel for 2D convolution with optimized block size and asynchronous streams
__global__ void conv2d_cuda_kernel_combined(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int chunk_N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = chunk_N * C_out * H_out * W_out;

    if (index >= total_threads) return;

    int w_out = index % W_out;
    int tmp = index / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    int group = c_out / (C_out / groups);
    int c_in_start = group * (C_in / groups);
    int c_in_end = c_in_start + (C_in / groups);

    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        for (int k_h = 0; k_h < K_h; ++k_h) {
            for (int k_w = 0; k_w < K_w; ++k_w) {
                int h_in = h_out * stride_h - padding_h + k_h * dilation_h;
                int w_in = w_out * stride_w - padding_w + k_w * dilation_w;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    int weight_idx = (((c_out * (C_in / groups) + (c_in - c_in_start)) * K_h + k_h) * K_w) + k_w;
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[output_idx] = value;
}

// Host function: optimized for both block size and streams
torch::Tensor conv2d_cuda_combined(
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

    int stride_h = stride[0];
    int stride_w = stride[1];
    int padding_h = padding[0];
    int padding_w = padding[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];

    int H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    int W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value().contiguous();
        bias_ptr = bias.data_ptr<float>();
    }

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    int num_streams = (N < 4) ? N : 4;
    int chunk_size = (N + num_streams - 1) / num_streams;

    size_t input_batch_size = C_in * H_in * W_in;
    size_t output_batch_size = C_out * H_out * W_out;

    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    int threads_per_block = 256;
    for (int i = 0; i < num_streams; ++i) {
        int batch_start = i * chunk_size;
        if (batch_start >= N) break;
        int chunk_N = std::min(chunk_size, static_cast<int>(N - batch_start));

        const float* input_chunk_ptr = input_ptr + batch_start * input_batch_size;
        float* output_chunk_ptr = output_ptr + batch_start * output_batch_size;

        int total_threads = chunk_N * C_out * H_out * W_out;
        int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

        conv2d_cuda_kernel_combined<<<blocks, threads_per_block, 0, streams[i]>>>(
            input_chunk_ptr,
            weight_ptr,
            bias_ptr,
            output_chunk_ptr,
            chunk_N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            K_h, K_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups
        );
    }

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda_combined, "Combined 2D convolution with optimized block size and streams (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1},
          py::arg("groups") = 1);
}