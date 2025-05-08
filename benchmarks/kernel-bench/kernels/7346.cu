/*
Combined CUDA kernel that leverages spatial tiling for improved memory coalescing
and inner-loop unrolling for reduced loop overhead, merging ideas from two implementations.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>

// Tile dimensions for mapping the spatial output
#define TILE_W 16
#define TILE_H 16

// Combined kernel: each thread computes one output element of (n, c_out, h_out, w_out)
__global__ void conv2d_combined_cuda_kernel(
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
    // Map grid.z as a combination of batch and output channel: blockIdx.z = n * C_out + c_out
    int block_z = blockIdx.z;
    int n = block_z / C_out;
    int c_out = block_z % C_out;

    // Map blockIdx.x and blockIdx.y for spatial tiling
    int h_out = blockIdx.y * TILE_H + threadIdx.y;
    int w_out = blockIdx.x * TILE_W + threadIdx.x;

    if (h_out >= H_out || w_out >= W_out) return;

    // Initialize output with bias value if provided
    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Determine input channel range for this group
    int group = c_out / (C_out / groups);
    int c_in_start = group * (C_in / groups);
    int c_in_end = c_in_start + (C_in / groups);

    // Loop over kernel spatial dimensions first to reuse computed indices
    for (int k_h_idx = 0; k_h_idx < K_h; ++k_h_idx) {
        int h_in = h_out * stride_h - padding_h + k_h_idx * dilation_h;
        if (h_in < 0 || h_in >= H_in) continue;
        for (int k_w_idx = 0; k_w_idx < K_w; ++k_w_idx) {
            int w_in = w_out * stride_w - padding_w + k_w_idx * dilation_w;
            if (w_in < 0 || w_in >= W_in) continue;
            
            // Process input channels in chunks of 4 with unrolling
            for (int c = c_in_start; c < c_in_end; c += 4) {
                int local_base = c - c_in_start;  // local channel offset within the group
                #pragma unroll
                for (int offset = 0; offset < 4; ++offset) {
                    if (c + offset < c_in_end) {
                        // Compute input index: ((n * C_in + channel) * H_in + h_in) * W_in + w_in
                        int input_idx = ((n * C_in + (c + offset)) * H_in + h_in) * W_in + w_in;
                        // Compute weight index: (((c_out * (C_in/groups) + local_channel) * K_h + k_h_idx) * K_w + k_w_idx)
                        int weight_idx = (((c_out * (C_in / groups) + (local_base + offset)) * K_h + k_h_idx) * K_w) + k_w_idx;
                        value += input[input_idx] * weight[weight_idx];
                    }
                } // end unroll over offset
            } // end loop over input channels
        } // end loop over kernel width
    } // end loop over kernel height

    // Write the computed value to the output tensor
    int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[output_idx] = value;
}

// Host function wrapping the combined CUDA kernel
torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    // Ensure tensors are contiguous and on CUDA
    input = input.contiguous();
    weight = weight.contiguous();
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value().contiguous();
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA if provided");
    }
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);

    int64_t C_out = weight.size(0);
    int64_t K_h = weight.size(2);
    int64_t K_w = weight.size(3);

    int stride_h = stride[0];
    int stride_w = stride[1];
    int padding_h = padding[0];
    int padding_w = padding[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];

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

    // Configure a 3D grid: x and y for spatial tiling; z for (n, c_out) pairs
    dim3 threads(TILE_W, TILE_H, 1);
    dim3 blocks(
        (W_out + TILE_W - 1) / TILE_W,
        (H_out + TILE_H - 1) / TILE_H,
        N * C_out
    );

    conv2d_combined_cuda_kernel<<<blocks, threads>>>(
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
        printf("Error in conv2d_combined_cuda_kernel: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda, "Combined 2D Convolution (CUDA) with tiling and unrolling",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1},
          py::arg("groups") = 1);
}
