#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device helper: Calculate flat index for input tensor
__device__ __forceinline__ int get_input_index(int n, int c, int h, int w, int C, int H, int W) {
    return n * C * H * W + c * H * W + h * W + w;
}

// Device helper: Calculate flat index for weight tensor (assuming groups == 1 and square kernel)
__device__ __forceinline__ int get_weight_index(int oc, int ic, int r, int s, int in_channels, int k) {
    return oc * in_channels * k * k + ic * k * k + r * k + s;
}

// Modular device function to compute the convolution for a single output pixel.
// Uses __ldg() for read-only cached loads.
__device__ __forceinline__ float compute_conv2d_pixel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int n,
    int oc,
    int out_y,
    int out_x,
    int in_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        #pragma unroll
        for (int r = 0; r < kernel_size; ++r) {
            #pragma unroll
            for (int s = 0; s < kernel_size; ++s) {
                int in_y = out_y * stride - padding + r * dilation;
                int in_x = out_x * stride - padding + s * dilation;
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int i_idx = get_input_index(n, ic, in_y, in_x, in_channels, in_height, in_width);
                    int w_idx = get_weight_index(oc, ic, r, s, in_channels, kernel_size);
                    float in_val = __ldg(&input[i_idx]);
                    float w_val = __ldg(&weight[w_idx]);
                    sum += in_val * w_val;
                }
            }
        }
    }
    if (bias != nullptr) {
        sum += __ldg(&bias[oc]);
    }
    return sum;
}

// CUDA kernel: each thread computes one output pixel
__global__ void modular_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    int total = batch * out_channels * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decode linear index into (n, oc, out_y, out_x)
    int out_x = idx % out_width;
    int tmp = idx / out_width;
    int out_y = tmp % out_height;
    tmp /= out_height;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;

    output[idx] = compute_conv2d_pixel(input, weight, bias, n, oc, out_y, out_x,
                                       in_channels, in_height, in_width,
                                       kernel_size, stride, padding, dilation);
}

// Host forward function to prepare dimensions and launch the kernel
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
    TORCH_CHECK(groups == 1, "Only groups==1 is supported.");

    int batch = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);  // square kernel assumed

    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width  = (in_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    int total = batch * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    modular_conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        dilation
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular CUDA forward function for 2D convolution using device functions");
}
