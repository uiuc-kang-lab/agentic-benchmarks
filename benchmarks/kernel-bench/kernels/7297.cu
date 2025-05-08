#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv2d_cuda_kernel_strided(
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
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int total_elements = N * C_out * H_out * W_out;
    
    // Each thread processes multiple elements with striding
    for (int idx = tid; idx < total_elements; idx += total_threads) {
        const int w_out = idx % W_out;
        int tmp = idx / W_out;
        const int h_out = tmp % H_out;
        tmp = tmp / H_out;
        const int c_out = tmp % C_out;
        const int n = tmp / C_out;

        // Initialize with bias if available
        float value = (bias != nullptr) ? bias[c_out] : 0.0f;

        const int group = c_out / (C_out / groups);
        const int c_in_start = group * (C_in / groups);
        const int c_in_end = c_in_start + (C_in / groups);

        // Pre-calculate input window position
        const int h_in_base = h_out * stride_h - padding_h;
        const int w_in_base = w_out * stride_w - padding_w;

        // Process input channels
        #pragma unroll 4
        for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
            const int input_c_offset = ((n * C_in + c_in) * H_in) * W_in;
            const int weight_c_offset = ((c_out * (C_in / groups) + (c_in - c_in_start)) * K_h) * K_w;

            // Process kernel window with unrolled loops where possible
            #pragma unroll
            for (int k_h = 0; k_h < K_h; ++k_h) {
                const int h_in = h_in_base + k_h * dilation_h;
                
                if (h_in >= 0 && h_in < H_in) {
                    const int input_h_offset = input_c_offset + h_in * W_in;
                    const int weight_h_offset = weight_c_offset + k_h * K_w;

                    #pragma unroll
                    for (int k_w = 0; k_w < K_w; ++k_w) {
                        const int w_in = w_in_base + k_w * dilation_w;
                        
                        if (w_in >= 0 && w_in < W_in) {
                            value += input[input_h_offset + w_in] * 
                                    weight[weight_h_offset + k_w];
                        }
                    }
                }
            }
        }

        // Write result
        output[idx] = value;
    }
}

torch::Tensor conv2d_cuda_strided(
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

    // Calculate optimal thread configuration
    const int threads_per_block = 256;
    const int total_elements = N * C_out * H_out * W_out;
    const int num_blocks = std::min(
        (total_elements + threads_per_block - 1) / threads_per_block,
        65535  // Maximum blocks per grid dimension
    );

    conv2d_cuda_kernel_strided<<<num_blocks, threads_per_block>>>(
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
    m.def("forward", &conv2d_cuda_strided, "Strided workload 2D convolution (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}