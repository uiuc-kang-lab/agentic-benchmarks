#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Constant memory for small kernel weights
__constant__ float const_weight[1024];  // Adjust size as needed

template<int KERNEL_H, int KERNEL_W>
__global__ void conv2d_cuda_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * C_out * H_out * W_out;

    if (index >= total_threads) return;

    // Compute output position
    const int w_out = index % W_out;
    int tmp = index / W_out;
    const int h_out = tmp % H_out;
    tmp = tmp / H_out;
    const int c_out = tmp % C_out;
    const int n = tmp / C_out;

    // Initialize accumulator with bias if present
    float sum = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Compute group information
    const int group = c_out / (C_out / groups);
    const int c_in_start = group * (C_in / groups);
    const int c_in_end = c_in_start + (C_in / groups);

    // Pre-compute input base indices
    const int input_batch_offset = n * C_in * H_in * W_in;
    const int weight_cout_offset = c_out * (C_in / groups) * KERNEL_H * KERNEL_W;

    // Manual unrolling for channel processing (process 4 channels at once)
    #pragma unroll
    for (int c_in = c_in_start; c_in < c_in_end; c_in += 4) {
        float4 partial_sums = {0.0f, 0.0f, 0.0f, 0.0f};
        
        // Manual unrolling of kernel height
        #pragma unroll
        for (int k_h = 0; k_h < KERNEL_H; ++k_h) {
            const int h_in = h_out * stride_h - padding_h + k_h * dilation_h;
            if (h_in >= 0 && h_in < H_in) {
                const int input_h_offset = input_batch_offset + h_in * W_in;
                const int weight_h_offset = weight_cout_offset + k_h * KERNEL_W;

                // Manual unrolling of kernel width
                #pragma unroll
                for (int k_w = 0; k_w < KERNEL_W; ++k_w) {
                    const int w_in = w_out * stride_w - padding_w + k_w * dilation_w;
                    if (w_in >= 0 && w_in < W_in) {
                        // Process 4 input channels simultaneously
                        #pragma unroll
                        for (int ci = 0; ci < 4 && (c_in + ci) < c_in_end; ++ci) {
                            const int input_idx = input_h_offset + (c_in + ci) * H_in * W_in + w_in;
                            const int weight_idx = weight_h_offset + ((c_in + ci - c_in_start) * KERNEL_H * KERNEL_W) + k_w;
                            
                            // Use constant memory for weights when possible
                            const float w_val = (weight_idx < 1024) ? const_weight[weight_idx] : weight[weight_idx - (c_in - c_in_start) * KERNEL_H * KERNEL_W];
                            partial_sums.x += (ci == 0) ? input[input_idx] * w_val : 0.0f;
                            partial_sums.y += (ci == 1) ? input[input_idx] * w_val : 0.0f;
                            partial_sums.z += (ci == 2) ? input[input_idx] * w_val : 0.0f;
                            partial_sums.w += (ci == 3) ? input[input_idx] * w_val : 0.0f;
                        }
                    }
                }
            }
        }
        sum += partial_sums.x + partial_sums.y + partial_sums.z + partial_sums.w;
    }

    // Write output with coalesced access
    output[((n * C_out + c_out) * H_out + h_out) * W_out + w_out] = sum;
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

    // Copy kernel weights to constant memory if small enough
    if (weight.numel() <= 1024) {
        cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), 
                          weight.numel() * sizeof(float));
    }

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        bias_ptr = bias_opt.value().contiguous().data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    const int threads = 256;
    const int blocks = (N * C_out * H_out * W_out + threads - 1) / threads;

    // Template specialization based on kernel size
    if (K_h == 3 && K_w == 3) {
        conv2d_cuda_kernel<3, 3><<<blocks, threads>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            N, C_in, H_in, W_in, C_out, H_out, W_out,
            stride_h, stride_w, padding_h, padding_w,
            dilation_h, dilation_w, groups);
    } else if (K_h == 5 && K_w == 5) {
        conv2d_cuda_kernel<5, 5><<<blocks, threads>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            N, C_in, H_in, W_in, C_out, H_out, W_out,
            stride_h, stride_w, padding_h, padding_w,
            dilation_h, dilation_w, groups);
    } else {
        conv2d_cuda_kernel<7, 7><<<blocks, threads>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            N, C_in, H_in, W_in, C_out, H_out, W_out,
            stride_h, stride_w, padding_h, padding_w,
            dilation_h, dilation_w, groups);
    }

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