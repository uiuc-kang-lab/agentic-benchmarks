#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define tile dimensions for the spatial output
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// CUDA kernel implementing 2D convolution with 2D tiled layout to ensure coalesced global memory accesses
__global__ void conv2d_cuda_kernel_coalesced(
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
    // Determine (n, c_out) from the third grid dimension
    int nc = blockIdx.z;  // linear index corresponds to a unique (n, c_out) pair
    int n = nc / C_out;
    int c_out = nc % C_out;

    // Compute the output spatial coordinates from 2D block indices
    int w_out_idx = blockIdx.x * TILE_WIDTH + threadIdx.x; // horizontal index in output
    int h_out_idx = blockIdx.y * TILE_HEIGHT + threadIdx.y;  // vertical index in output

    if (h_out_idx < H_out && w_out_idx < W_out) {
        // Initialize with bias if provided
        float value = (bias != nullptr) ? bias[c_out] : 0.0f;

        // Determine the group to which output channel belongs
        int group = c_out / (C_out / groups);
        int c_in_start = group * (C_in / groups);
        int c_in_end = c_in_start + (C_in / groups);

        // Loop over the corresponding input channels and kernel spatial dimensions
        for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
            for (int k_h = 0; k_h < K_h; ++k_h) {
                // Compute corresponding input row index
                int h_in = h_out_idx * stride_h - padding_h + k_h * dilation_h;
                if (h_in < 0 || h_in >= H_in) continue;
                for (int k_w = 0; k_w < K_w; ++k_w) {
                    // Compute corresponding input column index
                    int w_in = w_out_idx * stride_w - padding_w + k_w * dilation_w;
                    if (w_in < 0 || w_in >= W_in) continue;

                    // Compute flattened index for the input
                    int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    // Compute index into the weight tensor
                    int weight_idx = (((c_out * (C_in / groups) + (c_in - c_in_start)) * K_h + k_h) * K_w) + k_w;

                    // Accumulate the product
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }

        // Write the result to global memory (output is stored in channel-last order for spatial indices)
        int output_idx = ((n * C_out + c_out) * H_out + h_out_idx) * W_out + w_out_idx;
        output[output_idx] = value;
    }
}

// C++ interface to the PyTorch module

torch::Tensor conv2d_cuda_coalesced(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    // Ensure inputs are contiguous and on CUDA
    input = input.contiguous();
    weight = weight.contiguous();
    
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "Bias tensor must be on CUDA if provided");
    }

    // Extract dimensions
    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);

    int64_t C_out = weight.size(0);
    int64_t K_h = weight.size(2);
    int64_t K_w = weight.size(3);

    int64_t stride_h = stride[0];
    int64_t stride_w = stride[1];
    int64_t padding_h = padding[0];
    int64_t padding_w = padding[1];
    int64_t dilation_h = dilation[0];
    int64_t dilation_w = dilation[1];

    // Calculate output dimensions
    int64_t H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    int64_t W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;

    if (bias_opt.has_value()) {
        auto bias = bias_opt.value().contiguous();
        bias_ptr = bias.data_ptr<float>();
    }

    float* output_ptr = output.data_ptr<float>();

    // Define block and grid dimensions
    dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridDim((W_out + TILE_WIDTH - 1) / TILE_WIDTH,
                 (H_out + TILE_HEIGHT - 1) / TILE_HEIGHT,
                 N * C_out);  // each block in gridDim.z corresponds to one (n, c_out) pair

    conv2d_cuda_kernel_coalesced<<<gridDim, blockDim>>>(
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
        printf("Error in conv2d_cuda_kernel_coalesced: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda_coalesced, "2D convolution with coalesced memory accesses via tiled layout (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1},
          py::arg("groups") = 1);
}
