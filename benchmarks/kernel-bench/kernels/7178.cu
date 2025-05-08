#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized CUDA kernel using warp-level primitives for reduction
__global__ void conv2d_warp_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {

    int n = blockIdx.x;
    int oc = blockIdx.y;
    int out_y = blockIdx.z * blockDim.y + threadIdx.y;
    int out_x = threadIdx.x;

    if (out_y >= out_height || out_x >= out_width) return;

    float sum = 0.0f;

    const int warp_size = 32;
    int laneId = threadIdx.x % warp_size;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = laneId; kx < kernel_size; kx += warp_size) {
                int in_y = out_y * stride - padding + ky * dilation;
                int in_x = out_x * stride - padding + kx * dilation;

                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int input_idx = n * in_channels * in_height * in_width
                                  + ic * in_height * in_width
                                  + in_y * in_width + in_x;
                    int weight_idx = oc * in_channels * kernel_size * kernel_size
                                   + ic * kernel_size * kernel_size
                                   + ky * kernel_size + kx;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Warp reduction
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (laneId == 0) {
        if (bias) {
            sum += bias[oc];
        }
        int output_idx = n * out_channels * out_height * out_width
                       + oc * out_height * out_width
                       + out_y * out_width
                       + out_x;
        output[output_idx] = sum;
    }
}

// Host function that prepares the tensors and launches the CUDA kernel
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

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    // This implementation supports groups==1 only
    TORCH_CHECK(groups == 1, "Only groups==1 is supported in this optimized kernel");

    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    dim3 threads(32, 1);
    dim3 blocks(batch, out_channels, (out_height + threads.y - 1) / threads.y);

    conv2d_warp_optimized_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch, in_channels, out_channels,
        in_height, in_width, out_height, out_width,
        kernel_size, stride, padding, dilation);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-level optimized CUDA convolution implementation");
}
