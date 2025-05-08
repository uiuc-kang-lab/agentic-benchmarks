#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define tile dimensions for spatial tiling
#define TILE_W 8
#define TILE_H 8
#define TILE_D 4

// Optimized CUDA kernel for 3D transposed convolution using shared memory for the weight tile.
// Each block is assigned to a tile of the spatial dimensions (W, H, D) for a fixed (n, c_out) pair.
// The weight corresponding to the current output channel is first loaded into shared memory once,
// with a single __syncthreads() for consistency, reducing global memory accesses.

__global__ void conv_transposed_3d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int D_out, int H_out, int W_out,
    int groups,
    int out_channels_per_group,
    int input_channels_per_group,
    int num_tile_d  // number of depth tiles = ceil(D_out / TILE_D)
) {
    // Compute output spatial coordinates from grid and block indices.
    int out_w = blockIdx.x * TILE_W + threadIdx.x;
    int out_h = blockIdx.y * TILE_H + threadIdx.y;
    int block_d = blockIdx.z % num_tile_d;  // depth tile index within the (n, c_out) pair
    int nc_idx = blockIdx.z / num_tile_d;     // combined index for (n, c_out)
    int n = nc_idx / C_out;
    int c_out = nc_idx % C_out;
    int out_d = block_d * TILE_D + threadIdx.z;

    if (out_w >= W_out || out_h >= H_out || out_d >= D_out) return;

    // Determine group and channel indices
    int group = c_out / out_channels_per_group;
    int c_out_in_group = c_out % out_channels_per_group;

    // Allocate shared memory for the weight tile.
    // Each block loads the weight tensor for the corresponding (group, c_out_in_group) pair.
    // Weight shape: (C_in, out_channels_per_group, kD, kH, kW).
    // For the current group, only input_channels_per_group elements are used.
    extern __shared__ float sh_weight[]; // size = input_channels_per_group * (kD * kH * kW)
    int weight_tile_size = input_channels_per_group * kD * kH * kW;

    // Use the block's threads to cooperatively load the weight tile from global memory.
    int tid = threadIdx.z * (TILE_W * TILE_H) + threadIdx.y * TILE_W + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    for (int i = tid; i < weight_tile_size; i += blockSize) {
        int c = i / (kD * kH * kW);  // index among input channels in the group
        int k_idx = i % (kD * kH * kW);
        int actual_c_in = group * input_channels_per_group + c;
        int global_weight_index = ((actual_c_in * out_channels_per_group + c_out_in_group) * (kD * kH * kW)) + k_idx;
        sh_weight[i] = weight[global_weight_index];
    }
    __syncthreads();  // synchronize to ensure shared memory is fully populated

    // Initialize output value with bias if provided
    float out_val = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Loop over kernel dimensions and the input channels (within the group)
    for (int r = 0; r < kD; ++r) {
        int d_in_calc = out_d + pad_d - r;
        if (d_in_calc % stride_d != 0) continue;
        int d_in = d_in_calc / stride_d;
        if (d_in < 0 || d_in >= D_in) continue;
        for (int s = 0; s < kH; ++s) {
            int h_in_calc = out_h + pad_h - s;
            if (h_in_calc % stride_h != 0) continue;
            int h_in = h_in_calc / stride_h;
            if (h_in < 0 || h_in >= H_in) continue;
            for (int t = 0; t < kW; ++t) {
                int w_in_calc = out_w + pad_w - t;
                if (w_in_calc % stride_w != 0) continue;
                int w_in = w_in_calc / stride_w;
                if (w_in < 0 || w_in >= W_in) continue;

                // Accumulate over the input channels within the group
                for (int c = 0; c < input_channels_per_group; ++c) {
                    int input_channel = group * input_channels_per_group + c;
                    int input_index = (((n * C_in + input_channel) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                    float in_val = input[input_index];
                    int weight_offset = c * (kD * kH * kW) + (r * kH * kW + s * kW + t);
                    float w_val = sh_weight[weight_offset];
                    out_val += in_val * w_val;
                }
            }
        }
    }

    int output_index = ((((n * C_out + c_out) * D_out + out_d) * H_out + out_h) * W_out + out_w);
    output[output_index] = out_val;
}

// Forward function for the optimized 3D transposed convolution
// Input shape: (N, C_in, D_in, H_in, W_in)
// Weight shape: (C_in, out_channels_per_group, kD, kH, kW)
// Bias shape: (C_out) if provided
// Stride, Padding, and Output Padding are 3-element vectors
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
    // Input dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    // Kernel dimensions from the weight tensor
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    const int stride_d = stride[0];
    const int stride_h = stride[1];
    const int stride_w = stride[2];

    const int pad_d = padding[0];
    const int pad_h = padding[1];
    const int pad_w = padding[2];

    const int out_pad_d = output_padding[0];
    const int out_pad_h = output_padding[1];
    const int out_pad_w = output_padding[2];

    // Compute the output dimensions (assuming dilation = 1)
    const int D_out = (D_in - 1) * stride_d - 2 * pad_d + kD + out_pad_d;
    const int H_out = (H_in - 1) * stride_h - 2 * pad_h + kH + out_pad_h;
    const int W_out = (W_in - 1) * stride_w - 2 * pad_w + kW + out_pad_w;

    // Determine output channels; weight shape: (C_in, out_channels_per_group, kD, kH, kW)
    const int out_channels_per_group = weight.size(1);
    const int C_out = out_channels_per_group * groups;

    // Compute the number of input channels per group
    const int input_channels_per_group = C_in / groups;

    // Create output tensor
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    // Define block (tile) dimensions
    const int tileW = TILE_W;
    const int tileH = TILE_H;
    const int tileD = TILE_D;

    // Grid dimensions: tile across W and H, and for depth, each (n, c_out) pair is tiled along D.
    int grid_x = (W_out + tileW - 1) / tileW;
    int grid_y = (H_out + tileH - 1) / tileH;
    int num_tile_d = (D_out + tileD - 1) / tileD;
    int grid_z = N * C_out * num_tile_d;

    dim3 block(tileW, tileH, tileD);
    dim3 grid(grid_x, grid_y, grid_z);

    // Shared memory size for the weight tile
    int sharedMemSize = input_channels_per_group * kD * kH * kW * sizeof(float);

    // Get raw pointers
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    // Launch the CUDA kernel
    conv_transposed_3d_shared_kernel<<<grid, block, sharedMemSize, cudaStreamDefault>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        N, C_in, D_in, H_in, W_in,
        C_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        D_out, H_out, W_out,
        groups,
        out_channels_per_group,
        input_channels_per_group,
        num_tile_d
    );

    return output;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized ConvTranspose3d forward with shared weight tiling",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}
