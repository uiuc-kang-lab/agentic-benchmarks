#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Helper function to check valid indices (can be inlined by the compiler)
__device__ __forceinline__ bool is_valid(int h, int w, int H, int W) {
    return (h >= 0 && h < H) && (w >= 0 && w < W);
}

// Combined kernel that uses warp-aligned thread scheduling and optimized block occupancy
__global__ void conv2d_cuda_kernel_warp_optimized(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups
) {
    // Warp-based scheduling
    const int warp_size = 32;
    int lane_id = threadIdx.x % warp_size;
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warp_size;

    // Total number of output elements
    int total_outputs = N * C_out * H_out * W_out;
    int total_warps = (total_outputs + warp_size - 1) / warp_size;
    if (warp_id >= total_warps) return;

    // Each warp processes warp_size consecutive output elements
    int output_index = warp_id * warp_size + lane_id;
    if (output_index >= total_outputs) return;

    // Decode output_index into n, c_out, h_out, w_out
    int w_out = output_index % W_out;
    int tmp = output_index / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    // Initialize accumulator with bias if provided
    float out_val = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Determine group parameters
    int group_channels = C_in / groups;  // Number of input channels per group
    int c_in_group_start = (c_out / (C_out / groups)) * group_channels;

    // Compute top-left corner for the convolution window
    int h_in_origin = h_out * stride_h - pad_h;
    int w_in_origin = w_out * stride_w - pad_w;

    // Loop over input channels within the group
    for (int c_in = c_in_group_start; c_in < c_in_group_start + group_channels; ++c_in) {
        int input_channel_offset = ((n * C_in + c_in) * H_in);
        int weight_channel_offset = (((c_out * group_channels) + (c_in - c_in_group_start)) * K_h);

        #pragma unroll
        for (int k_h = 0; k_h < K_h; ++k_h) {
            int h_in = h_in_origin + k_h * dilation_h;
            if (h_in >= 0 && h_in < H_in) {
                int input_row_offset = input_channel_offset + h_in;
                int weight_row_offset = weight_channel_offset + k_h * K_w;

                #pragma unroll
                for (int k_w = 0; k_w < K_w; ++k_w) {
                    int w_in = w_in_origin + k_w * dilation_w;
                    if (w_in >= 0 && w_in < W_in) {
                        int input_index = input_row_offset * W_in + w_in;
                        int weight_index = weight_row_offset + k_w;
                        out_val += input[input_index] * weight[weight_index];
                    }
                }
            }
        }
    }

    // Write the computed value to the output tensor
    int output_base = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[output_base] = out_val;
}

// C++ interface for the PyTorch module using the combined kernel
torch::Tensor conv2d_cuda_warp_optimized(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    // Ensure tensors are contiguous and located on CUDA
    input = input.contiguous();
    weight = weight.contiguous();
    
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");

    const float* bias_ptr = nullptr;
    torch::Tensor bias;
    if (bias_opt.has_value()) {
        bias = bias_opt.value().contiguous();
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA if provided");
        bias_ptr = bias.data_ptr<float>();
    }

    // Input dimensions
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    // Weight dimensions: (C_out, C_in/groups, K_h, K_w)
    int C_out = weight.size(0);
    int K_h = weight.size(2);
    int K_w = weight.size(3);

    int stride_h = stride[0];
    int stride_w = stride[1];
    int pad_h = padding[0];
    int pad_w = padding[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];

    // Calculate output dimensions
    int H_out = (H_in + 2 * pad_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Determine kernel launch configuration using warps
    int total_elements = N * C_out * H_out * W_out;
    const int warp_size = 32;
    int total_warps = (total_elements + warp_size - 1) / warp_size;

    int threads_per_block = 256;  // 8 warps per block
    int total_threads = total_warps * warp_size;  // Ensure full warp occupancy
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    conv2d_cuda_kernel_warp_optimized<<<num_blocks, threads_per_block>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        groups
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in conv2d_cuda_kernel_warp_optimized: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda_warp_optimized, "Warp optimized 2D convolution (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1},
          py::arg("groups") = 1);
}
