#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>

#define WARP_SIZE 32

// CUDA kernel using warp-level primitives for reduction and shared memory optimization
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
    // Each warp computes one output pixel
    int total_outputs = N * C_out * H_out * W_out;
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = globalThreadId / WARP_SIZE;
    int laneId = globalThreadId % WARP_SIZE;

    if (warpId >= total_outputs) return;

    // Decode warpId into output coordinates
    int tmp = warpId;
    int w_out_idx = tmp % W_out;
    tmp /= W_out;
    int h_out_idx = tmp % H_out;
    tmp /= H_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    // Initialize result with bias if available
    float res = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Determine the group for the current output channel
    int group = c_out / (C_out / groups);
    int channels_per_group = C_in / groups;
    int group_start = group * channels_per_group;

    // Total number of elements to reduce over: for each channel in group, multiply by kernel size
    int total_elems = channels_per_group * K_h * K_w;

    float partial_sum = 0.0f;
    
    // Partition the summation among warp lanes
    for (int idx = laneId; idx < total_elems; idx += WARP_SIZE) {
        int local_c = idx / (K_h * K_w);
        int rem = idx % (K_h * K_w);
        int k_h = rem / K_w;
        int k_w = rem % K_w;
        int c_in = group_start + local_c;

        int h_in = h_out_idx * stride_h - padding_h + k_h * dilation_h;
        int w_in = w_out_idx * stride_w - padding_w + k_w * dilation_w;

        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
            int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
            int weight_idx = (((c_out * channels_per_group) + local_c) * K_h + k_h) * K_w + k_w;
            partial_sum += input[input_idx] * weight[weight_idx];
        }
    }

    // Warp-level reduction using __shfl_down_sync to sum partial sums
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    // Lane 0 writes the result
    if (laneId == 0) {
        int output_idx = ((n * C_out + c_out) * H_out + h_out_idx) * W_out + w_out_idx;
        output[output_idx] = res + partial_sum;
    }
}

// Host function wrapping the CUDA kernel

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

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value().contiguous();
        bias_ptr = bias.data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    int total_outputs = N * C_out * H_out * W_out;
    int numWarps = total_outputs; // one warp per output pixel
    int total_threads = numWarps * WARP_SIZE;

    int threads_per_block = 128; // multiple of warp size
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    conv2d_cuda_kernel<<<blocks, threads_per_block>>>(
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

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in conv2d_cuda_kernel: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda, "Custom 2D convolution with warp reduction (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1},
          py::arg("groups") = 1);
}
