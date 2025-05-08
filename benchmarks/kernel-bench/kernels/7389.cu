#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

// Using shared memory for weights to reduce global memory accesses
__global__ void conv2d_cuda_kernel_shared(
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

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (index >= total) return;

    // Decode the flat index into (n, c_out, h_out, w_out) coordinates
    int w_out = index % W_out;
    int tmp = index / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    // Initialize output value with bias (if provided) or zero
    float out_val = (bias != nullptr) ? bias[c_out] : 0.0f;

    int channels_per_group = C_in / groups;
    int group = c_out / (C_out / groups);
    int c_in_start = group * channels_per_group;

    // Load weights into shared memory
    int weight_size = C_out * channels_per_group * K_h * K_w;
    for (int i = threadIdx.x; i < weight_size; i += blockDim.x) {
        shared_weight[i] = weight[i];
    }
    __syncthreads();

    // Pre-compute base indices to reduce redundant calculations
    const int n_C_in = n * C_in;
    const int c_out_channels = c_out * channels_per_group;
    const int h_out_stride = h_out * stride_h - padding_h;
    const int w_out_stride = w_out * stride_w - padding_w;

    // Loop over the input channels within the group and over the kernel height/width
    for (int c_in = c_in_start; c_in < c_in_start + channels_per_group; c_in++) {
        const int n_C_in_c = n_C_in + c_in;
        const int weight_channel_offset = c_out_channels + (c_in - c_in_start);
        
        for (int kh = 0; kh < K_h; kh++) {
            const int h_in = h_out_stride + kh * dilation_h;
            // Branchless check: valid if 0 <= h_in < H_in
            const int valid_h = ((unsigned)h_in < (unsigned)H_in);
            const int n_C_in_c_H = n_C_in_c * H_in + h_in;
            const int weight_h_offset = weight_channel_offset * K_h + kh;
            
            if (valid_h) {  // Early exit if h_in is invalid
                for (int kw = 0; kw < K_w; kw++) {
                    const int w_in = w_out_stride + kw * dilation_w;
                    // Branchless check: valid if 0 <= w_in < W_in
                    const int valid_w = ((unsigned)w_in < (unsigned)W_in);
                    
                    if (valid_w) {  // Only compute if both h_in and w_in are valid
                        const int input_idx = n_C_in_c_H * W_in + w_in;
                        const int weight_idx = weight_h_offset * K_w + kw;
                        out_val += input[input_idx] * shared_weight[weight_idx];
                    }
                }
            }
            }
        }
    }

    int out_index = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[out_index] = out_val;
}

// C++ interface to the PyTorch module
torch::Tensor conv2d_cuda_shared(
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

    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(0);
    int K_h = weight.size(2);
    int K_w = weight.size(3);

    int stride_h = stride[0];
    int stride_w = stride[1];
    int padding_h = padding[0];
    int padding_w = padding[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];

    int H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    int W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    int total_threads = N * C_out * H_out * W_out;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = C_out * (C_in / groups) * K_h * K_w * sizeof(float);

    conv2d_cuda_kernel_shared<<<blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_opt.has_value() ? bias_opt.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in conv2d_cuda_kernel_shared: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda_shared, "Shared Memory 2D convolution kernel (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1},
          py::arg("groups") = 1);
}