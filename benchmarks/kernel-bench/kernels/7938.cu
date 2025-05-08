#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel for warp-level reduction
__device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void conv2d_kernel(const float *x, const float *weight, float *output, int N, int C, int H, int W, int kH, int kW, int outH, int outW, int stride, int padding) {
    int batch = blockIdx.x;
    int channel = blockIdx.y;
    int row = blockIdx.z * blockDim.y + threadIdx.y; // Output row
    int col = threadIdx.x; // Output column

    if (row < outH && col < outW) {
        float value = 0.0;
        for (int i = 0; i < kH; ++i) {
            for (int j = 0; j < kW; ++j) {
                int in_row = row * stride - padding + i;
                int in_col = -padding + col * stride + j;
                if (in_row >= 0 && in_row < H && in_col >= 0 && in_col < W) {
                    for (int c = 0; c < C; ++c) {
                        float x_val = x[batch * C * H * W + c * H * W + in_row * W + in_col];
                        float weight_val = weight[channel * C * kH * kW + c * kH * kW + i * kW + j];
                        value += x_val * weight_val;
                    }
                }
            }
        }
        // Perform warp-level reduction
        value = warpReduceSum(value);
        if (threadIdx.x % warpSize == 0) // Collect reduced values from lane 0 of each warp
            output[batch * outH * outW * C + channel * outH * outW + row * outW + col] = value;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    auto x_sizes = x.sizes();
    int N = x_sizes[0]; // Batch size
    int C = x_sizes[1]; // Number of input channels
    int H = x_sizes[2]; // Height of input
    int W = x_sizes[3]; // Width of input

    auto weight_sizes = weight.sizes();
    int kH = weight_sizes[2];
    int kW = weight_sizes[3];
    int outC = weight_sizes[0]; // Number of output channels

    int outH = (H + 2 * padding - dilation * (kH - 1) - 1) / stride + 1;
    int outW = (W + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;

    auto output = torch::empty({N, outC, outH, outW}, x.options());

    dim3 blocks(N, outC, (outH + 15) / 16); // Configuring the grid size
    dim3 threads(32, 16); // Each block has 512 threads

    conv2d_kernel<<<blocks, threads>>>(x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W, kH, kW, outH, outW, stride, padding);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with warp-level primitives");
}