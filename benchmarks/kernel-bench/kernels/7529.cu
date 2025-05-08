#include <torch/extension.h>
#include <vector>

// Optimized forward function implementing conv_transpose3d with even workload distribution
__global__ void convTranspose3D_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int K, int T, int R, int S,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int groups
) {
    int out_d = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_d < D && out_h < H && out_w < W) {
        for (int n = 0; n < N; ++n) {
            for (int g = 0; g < groups; ++g) {
                for (int k = 0; k < K; ++k) {
                    float sum = 0.0f;
                    for (int t = 0; t < T; ++t) {
                        for (int r = 0; r < R; ++r) {
                            for (int s = 0; s < S; ++s) {
                                int in_d = out_d * stride_d - pad_d + t;
                                int in_h = out_h * stride_h - pad_h + r;
                                int in_w = out_w * stride_w - pad_w + s;
                                if (in_d >= 0 && in_h >= 0 && in_w >= 0 && in_d < D && in_h < H && in_w < W) {
                                    int input_idx = ((n * C + g) * D + in_d) * H * W + in_h * W + in_w;
                                    int weight_idx = (((g * K + k) * T + t) * R + r) * S + s;
                                    sum += x[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                    int output_idx = ((n * K + k) * D + out_d) * H * W + out_h * W + out_w;
                    output[output_idx] = sum + (bias ? bias[k] : 0);
                }
            }
        }
    }
}

// Wrapper function to launch the kernel
void forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    torch::Tensor output,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto D = x.size(2);
    const auto H = x.size(3);
    const auto W = x.size(4);
    const auto K = weight.size(0);
    const auto T = weight.size(2);
    const auto R = weight.size(3);
    const auto S = weight.size(4);

    const dim3 threads(8, 8, 8);
    const dim3 blocks((D + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y, (W + threads.z - 1) / threads.z);

    convTranspose3D_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C, D, H, W, K, T, R, S,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        groups
    );
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized ConvTranspose3d forward function",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("output"),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}