#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define NUM_STREAMS 4
#define BLOCK_SIZE 256

__global__ void conv2d_cuda_kernel_stream(
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
    int groups,
    int stream_offset,
    int stream_elements
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= stream_elements) return;

    int global_tid = stream_offset + tid;
    
    int w_out = global_tid % W_out;
    int tmp = global_tid / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    int group = c_out / (C_out / groups);
    int c_in_start = group * (C_in / groups);
    int c_in_end = c_in_start + (C_in / groups);

    #pragma unroll 4
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        for (int k_h = 0; k_h < K_h; ++k_h) {
            int h_in = h_out * stride_h - padding_h + k_h * dilation_h;
            if (h_in >= 0 && h_in < H_in) {
                for (int k_w = 0; k_w < K_w; ++k_w) {
                    int w_in = w_out * stride_w - padding_w + k_w * dilation_w;
                    if (w_in >= 0 && w_in < W_in) {
                        int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                        int weight_idx = (((c_out * (C_in / groups) + (c_in - c_in_start)) * K_h + k_h) * K_w) + k_w;
                        value += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
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

    const int64_t stride_h = stride[0];
    const int64_t stride_w = stride[1];
    const int64_t padding_h = padding[0];
    const int64_t padding_w = padding[1];
    const int64_t dilation_h = dilation[0];
    const int64_t dilation_w = dilation[1];

    const int64_t H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    const int64_t W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value().contiguous();
        bias_ptr = bias.data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int total_elements = N * C_out * H_out * W_out;
    const int elements_per_stream = (total_elements + NUM_STREAMS - 1) / NUM_STREAMS;
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        const int stream_offset = i * elements_per_stream;
        const int stream_elements = min(elements_per_stream, total_elements - stream_offset);
        
        if (stream_elements <= 0) continue;

        const int threads = BLOCK_SIZE;
        const int blocks = (stream_elements + threads - 1) / threads;

        conv2d_cuda_kernel_stream<<<blocks, threads, 0, streams[i]>>>(
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
            groups,
            stream_offset,
            stream_elements
        );
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda, "Streamed 2D convolution (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}