#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void conv2d_cuda_kernel_warp(
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
    const unsigned int tid = threadIdx.x;
    const unsigned int wid = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    const unsigned int batch_idx = blockIdx.x / (C_out * H_out * W_out);
    const unsigned int feature_idx = (blockIdx.x / (H_out * W_out)) % C_out;
    const unsigned int h_out_idx = (blockIdx.x / W_out) % H_out;
    const unsigned int w_out_idx = blockIdx.x % W_out;
    
    // Early exit if out of bounds
    if (batch_idx >= N || feature_idx >= C_out || 
        h_out_idx >= H_out || w_out_idx >= W_out) return;

    const int group = feature_idx / (C_out / groups);
    const int c_in_start = group * (C_in / groups);
    const int c_in_end = c_in_start + (C_in / groups);
    
    // Initialize accumulator
    float partial_sum = 0.0f;
    
    // Each warp processes a subset of input channels
    for (int c_in = c_in_start + lane; c_in < c_in_end; c_in += WARP_SIZE) {
        for (int kh = 0; kh < K_h; kh++) {
            const int h_in = h_out_idx * stride_h - padding_h + kh * dilation_h;
            const bool valid_h = (h_in >= 0 && h_in < H_in);
            
            for (int kw = 0; kw < K_w; kw++) {
                const int w_in = w_out_idx * stride_w - padding_w + kw * dilation_w;
                const bool valid_w = (w_in >= 0 && w_in < W_in);
                
                if (valid_h && valid_w && c_in < c_in_end) {
                    const int input_idx = ((batch_idx * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    const int weight_idx = (((feature_idx * (C_in / groups) + (c_in - c_in_start)) * K_h + kh) * K_w) + kw;
                    partial_sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Warp reduction using shuffle operations
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }
    
    // First thread in each warp writes result
    if (lane == 0) {
        const int output_idx = ((batch_idx * C_out + feature_idx) * H_out + h_out_idx) * W_out + w_out_idx;
        float final_value = partial_sum;
        if (bias != nullptr && lane == 0) {
            final_value += bias[feature_idx];
        }
        output[output_idx] = final_value;
    }
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
    
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);
    auto C_out = weight.size(0);
    auto K_h = weight.size(2);
    auto K_w = weight.size(3);
    
    auto H_out = (H_in + 2 * padding[0] - dilation[0] * (K_h - 1) - 1) / stride[0] + 1;
    auto W_out = (W_in + 2 * padding[1] - dilation[1] * (K_w - 1) - 1) / stride[1] + 1;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    
    const int total_blocks = N * C_out * H_out * W_out;
    
    conv2d_cuda_kernel_warp<<<total_blocks, BLOCK_SIZE>>>(
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
    m.def("forward", &conv2d_cuda, "Warp-shuffle optimized 2D convolution (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1},
          py::arg("groups") = 1);
}