#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)

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
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + wid;
    
    const int total_outputs = N * C_out * H_out * W_out;
    const int total_warps = gridDim.x * WARPS_PER_BLOCK;
    const int channels_per_group = C_in / groups;
    
    for (int output_idx = global_warp_id; output_idx < total_outputs; output_idx += total_warps) {
        const int w_out = output_idx % W_out;
        int tmp = output_idx / W_out;
        const int h_out = tmp % H_out;
        tmp = tmp / H_out;
        const int c_out = tmp % C_out;
        const int n = tmp / C_out;

        const int group = c_out / (C_out / groups);
        const int c_in_start = group * channels_per_group;
        
        float partial_sum = 0.0f;
        
        for (int c_in = c_in_start + lane; c_in < c_in_start + channels_per_group; c_in += WARP_SIZE) {
            for (int kh = 0; kh < K_h; kh++) {
                const int h_in = h_out * stride_h - padding_h + kh * dilation_h;
                const bool valid_h = (unsigned)h_in < (unsigned)H_in;
                
                for (int kw = 0; kw < K_w; kw++) {
                    const int w_in = w_out * stride_w - padding_w + kw * dilation_w;
                    const bool valid_w = (unsigned)w_in < (unsigned)W_in;
                    
                    if (valid_h && valid_w && c_in < c_in_start + channels_per_group) {
                        const int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                        const int weight_idx = (((c_out * channels_per_group) + (c_in - c_in_start)) * K_h + kh) * K_w + kw;
                        partial_sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }

        if (lane == 0 && output_idx < total_outputs) {
            float final_sum = partial_sum;
            if (bias != nullptr) {
                final_sum += bias[c_out];
            }
            output[output_idx] = final_sum;
        }
    }
}

torch::Tensor conv2d_cuda_warp(
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

    const auto H_out = (H_in + 2 * padding[0] - dilation[0] * (K_h - 1) - 1) / stride[0] + 1;
    const auto W_out = (W_in + 2 * padding[1] - dilation[1] * (K_w - 1) - 1) / stride[1] + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const int total_elements = N * C_out * H_out * W_out;
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int grid_size = min(num_blocks, 65535);

    conv2d_cuda_kernel_warp<<<grid_size, BLOCK_SIZE>>>(
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
    m.def("forward", &conv2d_cuda_warp, "Warp-optimized 2D convolution (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}