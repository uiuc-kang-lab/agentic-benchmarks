#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel uses warp-level primitives to collaboratively compute each output pixel.
// Each warp is assigned one output element. The work (summing over channels and kernel window) is divided among the 32 lanes,
// and a warp-level reduction (using __shfl_down_sync) is performed to sum the partial results.

__global__ void conv2d_cuda_kernel_warp_reduce(
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
    // Each warp computes one output pixel.
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int total_output = N * C_out * H_out * W_out;
    if (warp_id >= total_output) return;

    // Map warp_id to output coordinates
    int out_index = warp_id;
    int w_out = out_index % W_out;
    int tmp = out_index / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    // Determine the input channel range for this output (based on groups)
    int group = c_out / (C_out / groups);
    int c_in_start = group * (C_in / groups);
    int c_in_end = c_in_start + (C_in / groups);
    int c_in_per_group = C_in / groups;

    // Output pixel corresponds to a convolution operating over a window in the input
    int h_in_origin = h_out * stride_h - padding_h;
    int w_in_origin = w_out * stride_w - padding_w;

    // Total number of multiply-accumulate iterations: over channels and kernel elements
    int total_iterations = (c_in_end - c_in_start) * (K_h * K_w);
    float sum = 0.0f;

    // Divide the work among the warp lanes
    for (int i = lane_id; i < total_iterations; i += 32) {
        int channel_offset = i / (K_h * K_w);  // offset in the channel group
        int rem = i % (K_h * K_w);
        int k_h = rem / K_w;
        int k_w = rem % K_w;
        int c_in = c_in_start + channel_offset;

        int h_in = h_in_origin + k_h * dilation_h;
        int w_in = w_in_origin + k_w * dilation_w;

        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
            int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
            int weight_idx = (((c_out * c_in_per_group + channel_offset) * K_h + k_h) * K_w) + k_w;
            sum += input[input_idx] * weight[weight_idx];
        }
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Lane 0 writes the final result
    if (lane_id == 0) {
        if (bias != nullptr) {
            sum += bias[c_out];
        }
        output[out_index] = sum;
    }
}

// C++ interface method
torch::Tensor conv2d_cuda_warp_reduce(
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

    int total_output = N * C_out * H_out * W_out;
    // Use 256 threads per block (8 warps per block)
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int blocks = (total_output + warps_per_block - 1) / warps_per_block;

    conv2d_cuda_kernel_warp_reduce<<<blocks, threads_per_block>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
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
        printf("Error in conv2d_cuda_kernel_warp_reduce: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda_warp_reduce, "Warp-level reduced 2D convolution (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1},
          py::arg("groups") = 1);
}
