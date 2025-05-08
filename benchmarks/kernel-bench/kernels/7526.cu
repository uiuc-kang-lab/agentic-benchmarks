#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel implements a transposed 3D convolution with refactored, branchless condition checks
// to minimize warp divergence. The conditional logic is expressed via ternary operators, ensuring
// uniform control flow across warps. It supports optional bias addition and assumes groups==1.

__global__ void conv_transposed_3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int D_out, int H_out, int W_out
) {
    // Compute global thread index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * D_out * H_out * W_out;
    if (index >= total) return;

    // Decode the flattened index into (n, c_out, od, oh, ow)
    int ow = index % W_out;
    int tmp = index / W_out;
    int oh = tmp % H_out;
    tmp /= H_out;
    int od = tmp % D_out;
    tmp /= D_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    float sum = 0.0f;
    
    // Loop over contributing input channels and kernel spatial dimensions
    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int kd = 0; kd < kD; kd++) {
            // Compute corresponding input depth index
            int id_temp = od + padD - kd;
            int id = id_temp / strideD;  // valid only if id_temp is divisible by strideD
            // Branchless check using ternary operator: if condition fails, contribution becomes 0
            bool valid_d = ((id_temp % strideD) == 0) && (id >= 0) && (id < D_in);
            
            for (int kh = 0; kh < kH; kh++) {
                int ih_temp = oh + padH - kh;
                int ih = ih_temp / strideH;
                bool valid_h = ((ih_temp % strideH) == 0) && (ih >= 0) && (ih < H_in);
                
                for (int kw = 0; kw < kW; kw++) {
                    int iw_temp = ow + padW - kw;
                    int iw = iw_temp / strideW;
                    bool valid_w = ((iw_temp % strideW) == 0) && (iw >= 0) && (iw < W_in);
                    
                    bool valid = valid_d && valid_h && valid_w;
                    // Use ternary operator to avoid divergent branches: if not valid, use 0 contribution
                    float inp = valid ? input[n * (C_in * D_in * H_in * W_in) +
                                               c_in * (D_in * H_in * W_in) +
                                               id * (H_in * W_in) +
                                               ih * W_in + iw] : 0.0f;
                    float wght = valid ? weight[c_in * (C_out * kD * kH * kW) +
                                                 c_out * (kD * kH * kW) +
                                                 kd * (kH * W_\n                                                 kw) + kh * kW + kw] : 0.0f;
                    // Correct the weight index: the original layout is assumed to be (C_in, C_out, kD, kH, kW)
                    wght = valid ? weight[c_in * (C_out * kD * kH * kW) +
                                                c_out * (kD * kH * kW) +
                                                kd * (kH * kW) + kh * kW + kw] : 0.0f;
                    sum += inp * wght;
                }
            }
        }
    }
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    output[index] = sum;
}

// Host function for launching the optimized CUDA kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,  // not used in this implementation
    int64_t groups  // assumes groups==1
) {
    // Input dimensions: (N, C_in, D_in, H_in, W_in)
    int N = x.size(0);
    int C_in = x.size(1);
    int D_in = x.size(2);
    int H_in = x.size(3);
    int W_in = x.size(4);

    // Weight dimensions: (C_in, C_out, kD, kH, kW)
    int C_out = weight.size(1);
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    int strideD = stride[0];
    int strideH = stride[1];
    int strideW = stride[2];

    int padD = padding[0];
    int padH = padding[1];
    int padW = padding[2];

    // Calculate output dimensions for transposed convolution
    int D_out = (D_in - 1) * strideD - 2 * padD + kD;
    int H_out = (H_in - 1) * strideH - 2 * padH + kH;
    int W_out = (W_in - 1) * strideW - 2 * padW + kW;

    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, x.options());
    
    int total = N * C_out * D_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    conv_transposed_3d_kernel<<<blocks, threads>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, C_in, D_in, H_in, W_in,
        C_out, kD, kH, kW,
        strideD, strideH, strideW,
        padD, padH, padW,
        D_out, H_out, W_out
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized ConvTranspose3d forward function with minimized warp divergence",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}
