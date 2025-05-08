#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device helper: Fetch a value from the input tensor
__device__ inline float get_input_value(const float* input, int n, int c, int h, int w, int C, int H, int W) {
    return input[n * C * H * W + c * H * W + h * W + w];
}

// Device helper: Fetch a value from the weight tensor
__device__ inline float get_weight_value(const float* weight, int m, int c, int r, int s, int in_channels_per_group, int k) {
    return weight[m * (in_channels_per_group * k * k) + c * (k * k) + r * k + s];
}

// Modular device function to compute convolution for one output pixel
__device__ float compute_conv2d_pixel(
    const float* input,
    const float* weight,
    const float* bias,
    int n, int m, int i, int j,
    int N, int C, int H, int W,
    int OC, int k, int stride, int padding, int dilation, int groups) {
    float sum = (bias != nullptr) ? bias[m] : 0.0f;
    int in_channels_per_group = C / groups;
    int group_id = m / (OC / groups);
    
    for (int c = 0; c < in_channels_per_group; c++) {
        int input_channel = group_id * in_channels_per_group + c;
        for (int r = 0; r < k; r++) {
            int in_h = i * stride - padding + r * dilation;
            if (in_h < 0 || in_h >= H) continue;
            for (int s = 0; s < k; s++) {
                int in_w = j * stride - padding + s * dilation;
                if (in_w < 0 || in_w >= W) continue;
                float in_val = get_input_value(input, n, input_channel, in_h, in_w, C, H, W);
                float w_val = get_weight_value(weight, m, c, r, s, in_channels_per_group, k);
                sum += in_val * w_val;
            }
        }
    }
    return sum;
}

// Global kernel launching the modularized convolution
__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int OC, int outH, int outW,
    int k, int stride, int padding, int dilation, int groups) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OC * outH * outW;
    if (index >= total) return;

    // Decode linear index into (n, m, i, j)
    int j = index % outW;
    int temp = index / outW;
    int i = temp % outH;
    temp /= outH;
    int m = temp % OC;
    int n = temp / OC;

    output[index] = compute_conv2d_pixel(input, weight, bias, n, m, i, j, N, C, H, W, OC, k, stride, padding, dilation, groups);
}

// Host function: Compute output dimensions and launch kernel
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

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int OC = weight.size(0); // number of output channels
    int k = weight.size(2);  // kernel height (assuming square kernel, so weight.size(2)==weight.size(3))
    
    int outH = (H + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    int outW = (W + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

    auto output = torch::zeros({N, OC, outH, outW}, x.options());
    
    int total = N * OC * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    conv2d_kernel<<<blocks, threads>>>(input_ptr, weight_ptr, bias_ptr, output_ptr,
                                         N, C, H, W, OC, outH, outW, k, stride, padding, dilation, groups);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for modularized 2D convolution");
}
