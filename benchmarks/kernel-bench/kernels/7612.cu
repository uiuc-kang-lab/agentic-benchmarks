#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel that evenly distributes output elements using a 3D grid mapping
// Each thread computes exactly one output element corresponding to (n, c_out, d, h, w)
__global__ void conv_transposed_3d_evenload_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int groups
) {
    // Compute output spatial coordinate from block and thread indices
    // Use 3D grid: grid.x covers width, grid.y covers height, grid.z covers (n, c_out, d)
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    // Decode the third grid dimension into n, c_out, d
    int linear = blockIdx.z; // ranges from 0 to (N * C_out * D_out - 1)
    int d = linear % D_out;
    int tmp = linear / D_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    // Check boundaries for width and height
    if (w >= W_out || h >= H_out)
        return;

    // Determine grouping info
    int output_channels_per_group = C_out / groups;
    int input_channels_per_group = C_in / groups;
    int group = c_out / output_channels_per_group;
    int c_out_in_group = c_out - group * output_channels_per_group;

    // Initialize output value with bias if provided
    float out_val = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Loop over kernel depth, height, width
    for (int r = 0; r < kD; ++r) {
        int d_in_calc = d + pad_d - r;
        if (d_in_calc % stride_d != 0) continue;
        int d_in = d_in_calc / stride_d;
        if (d_in < 0 || d_in >= D_in) continue;
        
        for (int s = 0; s < kH; ++s) {
            int h_in_calc = h + pad_h - s;
            if (h_in_calc % stride_h != 0) continue;
            int h_in = h_in_calc / stride_h;
            if (h_in < 0 || h_in >= H_in) continue;
            
            for (int t = 0; t < kW; ++t) {
                int w_in_calc = w + pad_w - t;
                if (w_in_calc % stride_w != 0) continue;
                int w_in = w_in_calc / stride_w;
                if (w_in < 0 || w_in >= W_in) continue;
                
                // Sum over the input channels in the group
                for (int c = 0; c < input_channels_per_group; ++c) {
                    int actual_c_in = group * input_channels_per_group + c;
                    int input_index = (((n * C_in + actual_c_in) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                    float in_val = input[input_index];
                    
                    // Weight layout: (C_in, C_out/groups, kD, kH, kW)
                    int weight_index = ((actual_c_in * output_channels_per_group + c_out_in_group) * (kD * kH * kW))
                                       + (r * kH * kW + s * kW + t);
                    float w_val = weight[weight_index];
                    
                    out_val += in_val * w_val;
                }
            }
        }
    }

    // Write the computed output element to the proper location
    int output_index = ((((n * C_out + c_out) * D_out) + d) * H_out + h) * W_out + w;
    output[output_index] = out_val;
}

// Forward function for the optimized conv_transposed3d with even workload distribution
// Input shape: (N, C_in, D_in, H_in, W_in)
// Weight shape: (C_in, C_out/groups, kD, kH, kW)
// Bias shape: (C_out) or can be nullptr
// Stride, Padding, Output Padding are provided as 3-element vectors
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

    // Kernel dimensions from weight tensor
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    // Get stride and padding values
    const int stride_d = stride[0];
    const int stride_h = stride[1];
    const int stride_w = stride[2];

    const int pad_d = padding[0];
    const int pad_h = padding[1];
    const int pad_w = padding[2];

    const int out_pad_d = output_padding[0];
    const int out_pad_h = output_padding[1];
    const int out_pad_w = output_padding[2];

    // Compute output dimensions (assuming dilation=1):
    // D_out = (D_in - 1) * stride_d - 2 * pad_d + kD + out_pad_d, etc.
    const int D_out = (D_in - 1) * stride_d - 2 * pad_d + kD + out_pad_d;
    const int H_out = (H_in - 1) * stride_h - 2 * pad_h + kH + out_pad_h;
    const int W_out = (W_in - 1) * stride_w - 2 * pad_w + kW + out_pad_w;

    // Compute output channels from weight dimensions
    const int output_channels_per_group = weight.size(1);
    const int C_out = output_channels_per_group * groups;

    // Create output tensor
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    // Define block and grid dimensions for the 3D mapping
    const int TILE_W = 16;
    const int TILE_H = 16;
    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid((W_out + TILE_W - 1) / TILE_W, (H_out + TILE_H - 1) / TILE_H, N * C_out * D_out);

    // Get raw pointers
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    // Launch the CUDA kernel with even workload distribution
    conv_transposed_3d_evenload_kernel<<<grid, block, 0, cudaStreamDefault>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
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
    m.def("forward", &forward, "Optimized ConvTranspose3d forward with even workload distribution",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}
