#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_NUM_THREADS 1024

// Helper function to compute number of CUDA blocks needed
inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// Kernel for 3D transposed convolution with minimized warp divergence
// This kernel uses arithmetic masks to replace conditional branches in the inner loops,
// ensuring uniform control flow across warps.
__global__ void conv_transposed_3d_kernel(
    const float * __restrict__ input,
    const float * __restrict__ weight,
    float * __restrict__ output,
    int N,
    int C_in,
    int D_in,
    int H_in,
    int W_in,
    int C_out,
    int kD,
    int kH,
    int kW,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int D_out,
    int H_out,
    int W_out,
    int groups
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * D_out * H_out * W_out;
    if (index < total) {
        // Decode global output index into n, c_out, d, h, w
        int w = index % W_out;
        int h = (index / W_out) % H_out;
        int d = (index / (W_out * H_out)) % D_out;
        int c_out = (index / (W_out * H_out * D_out)) % C_out;
        int n = index / (W_out * H_out * D_out * C_out);

        float sum = 0.0f;

        // For grouped convolution, determine group and local channel indices
        int group_out_channels = C_out / groups;       // number of output channels per group
        int group_in_channels = C_in / groups;           // number of input channels per group
        int g = c_out / group_out_channels;              // group index
        int c_out_local = c_out - g * group_out_channels;  
        int group_weight_offset = g * (group_in_channels * group_out_channels * kD * kH * kW);

        // Loop over the input channels within the current group
        for (int ci = 0; ci < group_in_channels; ci++) {
            int actual_c_in = g * group_in_channels + ci;
            // Loop over kernel depth, height, and width
            for (int kd = 0; kd < kD; kd++) {
                // Compute the corresponding input depth index
                int t = d + pad_d - kd;
                int valid_d = ((t % stride_d) == 0);
                int id = valid_d ? t / stride_d : 0;  // dummy value when not valid
                for (int kh = 0; kh < kH; kh++) {
                    int s = h + pad_h - kh;
                    int valid_h = ((s % stride_h) == 0);
                    int ih = valid_h ? s / stride_h : 0;
                    for (int kw = 0; kw < kW; kw++) {
                        int r = w + pad_w - kw;
                        int valid_w = ((r % stride_w) == 0);
                        int iw = valid_w ? r / stride_w : 0;

                        // Combine validity flags into a mask (1.0 if all conditions are met, 0.0 otherwise).
                        float mask = float(valid_d && valid_h && valid_w &&
                                             (id >= 0 && id < D_in) &&
                                             (ih >= 0 && ih < H_in) &&
                                             (iw >= 0 && iw < W_in));

                        int input_index = (((n * C_in + actual_c_in) * D_in + id) * H_in + ih) * W_in + iw;
                        int weight_index = group_weight_offset + (((ci * group_out_channels + c_out_local) * kD + kd) * kH + kh) * kW + kw;

                        sum += mask * input[input_index] * weight[weight_index];
                    }
                }
            }
        }
        int output_index = (((n * C_out + c_out) * D_out + d) * H_out + h) * W_out + w;
        output[output_index] = sum;
    }
}

// Forward function invoked from Python
// x: Input tensor of shape (N, C_in, D_in, H_in, W_in)
// weight: Convolution kernel of shape (C_in, C_out/groups, kD, kH, kW)
// bias: Optional bias tensor
// stride, padding, output_padding: Convolution parameters
// groups: Number of groups for grouped convolution

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Input dimensions
    int N = x.size(0);
    int C_in = x.size(1);
    int D_in = x.size(2);
    int H_in = x.size(3);
    int W_in = x.size(4);

    // Kernel dimensions
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);
    // For conv_transpose3d, weight has shape (C_in, C_out/groups, kD, kH, kW)
    // Thus total output channels C_out = (C_out/groups) * groups
    int C_out = weight.size(1) * groups;

    int stride_d = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];
    int pad_d = padding[0];
    int pad_h = padding[1];
    int pad_w = padding[2];
    int outpad_d = output_padding[0];
    int outpad_h = output_padding[1];
    int outpad_w = output_padding[2];

    // Calculate output dimensions using the transposed convolution formula
    int D_out = (D_in - 1) * stride_d - 2 * pad_d + kD + outpad_d;
    int H_out = (H_in - 1) * stride_h - 2 * pad_h + kH + outpad_h;
    int W_out = (W_in - 1) * stride_w - 2 * pad_w + kW + outpad_w;

    auto options = x.options();
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, options);

    int total = N * C_out * D_out * H_out * W_out;
    int threads = CUDA_NUM_THREADS;
    int blocks = GET_BLOCKS(total);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    conv_transposed_3d_kernel<<<blocks, threads>>>(
        input_ptr,
        weight_ptr,
        output_ptr,
        N, C_in, D_in, H_in, W_in,
        C_out, kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        D_out, H_out, W_out,
        groups
    );
    cudaDeviceSynchronize();

    // If a bias is provided, add it uniformly to the output tensor
    if (bias.has_value() && bias.value().defined()) {
        auto b = bias.value();
        output += b.view({1, C_out, 1, 1, 1});
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized ConvTranspose3d forward with minimized warp divergence",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}
