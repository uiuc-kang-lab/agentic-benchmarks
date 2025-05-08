#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template<int KERNEL_H, int KERNEL_W>
__device__ __forceinline__ float compute_conv_value_unrolled(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const int n, const int c_in,
    const int h_out, const int w_out,
    const int H_in, const int W_in,
    const int stride_h, const int stride_w,
    const int padding_h, const int padding_w,
    const int dilation_h, const int dilation_w,
    const int input_offset, const int weight_offset
) {
    float sum = 0.0f;
    const int h_in_base = h_out * stride_h - padding_h;
    const int w_in_base = w_out * stride_w - padding_w;

    #pragma unroll
    for (int kh = 0; kh < KERNEL_H; ++kh) {
        const int h_in = h_in_base + kh * dilation_h;
        const bool h_valid = (h_in >= 0) && (h_in < H_in);
        
        if (h_valid) {
            const int in_h_offset = input_offset + h_in * W_in;
            const int weight_h_offset = weight_offset + kh * KERNEL_W;
            
            #pragma unroll
            for (int kw = 0; kw < KERNEL_W; ++kw) {
                const int w_in = w_in_base + kw * dilation_w;
                if (w_in >= 0 && w_in < W_in) {
                    sum += input[in_h_offset + w_in] * weight[weight_h_offset + kw];
                }
            }
        }
    }
    return sum;
}

template<int KERNEL_H, int KERNEL_W>
__global__ void conv2d_cuda_kernel_unrolled(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out,
    const int stride_h, const int stride_w,
    const int padding_h, const int padding_w,
    const int dilation_h, const int dilation_w,
    const int groups
) {
    const int warp_size = 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = N * C_out * H_out * W_out;
    
    // Process multiple elements per thread for better utilization
    #pragma unroll 4
    for (int idx = tid; idx < total_threads; idx += blockDim.x * gridDim.x) {
        const int w_out = idx % W_out;
        int tmp = idx / W_out;
        const int h_out = tmp % H_out;
        tmp /= H_out;
        const int c_out = tmp % C_out;
        const int n = tmp / C_out;

        const int group = c_out / (C_out / groups);
        const int c_in_start = group * (C_in / groups);
        const int c_in_end = c_in_start + (C_in / groups);

        float value = bias ? bias[c_out] : 0.0f;

        #pragma unroll 4
        for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
            const int input_offset = ((n * C_in + c_in) * H_in);
            const int weight_offset = ((c_out * (C_in / groups) + (c_in - c_in_start)) * KERNEL_H);

            value += compute_conv_value_unrolled<KERNEL_H, KERNEL_W>(
                input, weight,
                n, c_in, h_out, w_out,
                H_in, W_in,
                stride_h, stride_w,
                padding_h, padding_w,
                dilation_h, dilation_w,
                input_offset, weight_offset
            );
        }
        
        output[idx] = value;
    }
}

torch::Tensor conv2d_cuda_unrolled(
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
    const int num_blocks = std::min(65535, (total_elements + threads_per_block - 1) / threads_per_block);

    if (K_h == 3 && K_w == 3) {
        conv2d_cuda_kernel_unrolled<3, 3><<<num_blocks, threads_per_block>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups
        );
    } else if (K_h == 5 && K_w == 5) {
        conv2d_cuda_kernel_unrolled<5, 5><<<num_blocks, threads_per_block>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups
        );
    } else {
        conv2d_cuda_kernel_unrolled<7, 7><<<num_blocks, threads_per_block>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda_unrolled, "Unrolled 2D convolution (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}