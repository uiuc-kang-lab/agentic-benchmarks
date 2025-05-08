#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Define tile dimensions for output spatial tiling
#define TILE_W 8
#define TILE_H 8
#define TILE_D 4

// CUDA kernel using shared memory for weight reuse
// This kernel assumes that each block works on a fixed batch index (n) and fixed output channel (c_out),
// processing a tile of output spatial locations (w, h, d).
// It loads the corresponding weight slice for the fixed c_out (i.e. for its group) into shared memory,
// and then each thread computes its output by summing contributions from the input and the shared weight.

__global__ void conv_transposed_3d_cuda_kernel_shared(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N,
    int C_in,
    int D_in,
    int H_in,
    int W_in,
    int C_out,
    int D_out,
    int H_out,
    int W_out,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int groups
) {
    // Calculate number of tiles in depth dimension for output
    int grid_d = (D_out + TILE_D - 1) / TILE_D;

    // Decode blockIdx.z to obtain the tile index in depth and also the combined index for n and c_out.
    // We pack (n, c_out, tile_d_index) into blockIdx.z as follows:
    // let temp = blockIdx.z, then tile_d_idx = temp % grid_d; and nc = temp / grid_d;
    // where nc encodes n and c_out: n = nc / C_out, c_out = nc % C_out.
    int tile_d_idx = blockIdx.z % grid_d;
    int nc = blockIdx.z / grid_d;
    int n = nc / C_out;
    int c_out = nc % C_out;

    // Determine global output coordinates from the block indices and thread indices
    int out_w = blockIdx.x * TILE_W + threadIdx.x; // along width
    int out_h = blockIdx.y * TILE_H + threadIdx.y; // along height
    int out_d = tile_d_idx * TILE_D + threadIdx.z;  // along depth

    // Check bounds
    if (out_w >= W_out || out_h >= H_out || out_d >= D_out) {
        return;
    }

    // Determine group info
    int output_channels_per_group = C_out / groups;  // weight shape: (C_in, C_out/groups, kD, kH, kW)
    int group = c_out / output_channels_per_group;
    int c_out_in_group = c_out % output_channels_per_group;
    int in_channels_grp = C_in / groups;  // number of input channels per group
    int kernelVol = kD * kH * kW;

    // Allocate shared memory for the weight tile for this (group, c_out_in_group).
    // Size = in_channels_grp * kernelVol floats
    extern __shared__ float sh_weight[];
    int weight_tile_size = in_channels_grp * kernelVol;

    // Compute linear thread id within the block
    int tid = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y * blockDim.z;

    // Cooperative loading of the weight tile from global memory into shared memory
    // Weight tensor shape: (C_in, output_channels_per_group, kD, kH, kW)
    // For the current block, we need to load for input channels c in [0, in_channels_grp) corresponding to global channel index = group * in_channels_grp + c,
    // and for the fixed output channel index c_out_in_group.
    for (int i = tid; i < weight_tile_size; i += block_threads) {
        int c = i / kernelVol;       // index within input channels of the group
        int offset = i % kernelVol;    // kernel offset index
        int global_weight_idx = (((group * in_channels_grp + c) * output_channels_per_group + c_out_in_group) * kernelVol) + offset;
        sh_weight[i] = weight[global_weight_idx];
    }
    __syncthreads();

    // Initialize the accumulation with bias if available
    float sum = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Loop over kernel dimensions
    // For each output spatial location, compute corresponding input indices per kernel offset.
    for (int r = 0; r < kD; r++) {
        int tmp_d = out_d + pad_d + r - (kD - 1);
        if (tmp_d < 0 || (tmp_d % stride_d) != 0) continue;
        int d_in = tmp_d / stride_d;
        if (d_in < 0 || d_in >= D_in) continue;
        for (int s = 0; s < kH; s++) {
            int tmp_h = out_h + pad_h - s;
            if (tmp_h < 0 || (tmp_h % stride_h) != 0) continue;
            int h_in = tmp_h / stride_h;
            if (h_in < 0 || h_in >= H_in) continue;
            for (int t = 0; t < kW; t++) {
                int tmp_w = out_w + pad_w - t;
                if (tmp_w < 0 || (tmp_w % stride_w) != 0) continue;
                int w_in = tmp_w / stride_w;
                if (w_in < 0 || w_in >= W_in) continue;
                int kernel_offset = r * (kH * kW) + s * kW + t;
                // Accumulate over input channels in the group
                for (int c = 0; c < in_channels_grp; c++) {
                    int input_channel = group * in_channels_grp + c;
                    int input_index = (((n * C_in + input_channel) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                    float in_val = input[input_index];
                    float w_val = sh_weight[c * kernelVol + kernel_offset];
                    sum += in_val * w_val;
                }
            }
        }
    }

    // Write the computed output value to the output tensor
    int out_index = (((n * C_out + c_out) * D_out + out_d) * H_out + out_h) * W_out + out_w;
    output[out_index] = sum;
}

// Forward function for ConvTranspose3d using the shared memory optimized kernel
// Input: shape (N, C_in, D_in, H_in, W_in)
// Weight: shape (C_in, C_out/groups, kD, kH, kW)
// Bias: shape (C_out) or nullptr
// Stride, Padding, and Output Padding are vectors of 3 elements each
// Groups: number of groups

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Get input dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    // Get kernel dimensions from weight tensor
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    // Stride and padding
    const int stride_d = stride[0];
    const int stride_h = stride[1];
    const int stride_w = stride[2];

    const int pad_d = padding[0];
    const int pad_h = padding[1];
    const int pad_w = padding[2];

    const int out_pad_d = output_padding[0];
    const int out_pad_h = output_padding[1];
    const int out_pad_w = output_padding[2];

    // Compute output dimensions (assuming dilation = 1):
    const int D_out = (D_in - 1) * stride_d - 2 * pad_d + kD + out_pad_d;
    const int H_out = (H_in - 1) * stride_h - 2 * pad_h + kH + out_pad_h;
    const int W_out = (W_in - 1) * stride_w - 2 * pad_w + kW + out_pad_w;

    // Calculate C_out from weight shape: weight shape is (C_in, C_out/groups, kD, kH, kW)
    const int output_channels_per_group = weight.size(1);
    const int C_out = output_channels_per_group * groups;

    // Create output tensor
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    // Set up block and grid dimensions for the kernel
    dim3 block(TILE_W, TILE_H, TILE_D);
    int grid_x = (W_out + TILE_W - 1) / TILE_W;
    int grid_y = (H_out + TILE_H - 1) / TILE_H;
    int grid_d = (D_out + TILE_D - 1) / TILE_D;
    // We pack the (n, c_out) dimensions into grid.z along with the depth tiling
    int grid_z = N * C_out * grid_d;
    dim3 grid(grid_x, grid_y, grid_z);

    // Calculate shared memory size: weight tile for one (group, c_out) slice
    int in_channels_grp = C_in / groups;
    int kernelVol = kD * kH * kW;
    size_t sharedMemSize = in_channels_grp * kernelVol * sizeof(float);

    // Launch the kernel
    conv_transposed_3d_cuda_kernel_shared<<<grid, block, sharedMemSize, cudaStreamDefault>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        (bias.has_value() && bias.value().defined()) ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        groups
    );

    return output;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward function with shared memory optimization",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}
