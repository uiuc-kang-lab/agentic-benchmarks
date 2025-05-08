#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel avoids using shared memory for weights and instead leverages the read-only data cache via __ldg, thereby reducing synchronization overhead.
// thereby eliminating the need for a __syncthreads() barrier after loading shared memory. This reduces synchronization overhead
// and can improve performance on architectures like the NVIDIA H100 with CUDA 12.2, while still producing correct results.

__global__ void conv2d_ldg_kernel(const float* __restrict__ input,
                                   const float* __restrict__ weight,
                                   const float* __restrict__ bias,
                                   float* __restrict__ output,
                                   int batch_size,
                                   int in_channels,
                                   int in_h,
                                   int in_w,
                                   int out_channels,
                                   int kernel_size,
                                   int out_h,
                                   int out_w,
                                   int stride,
                                   int padding) {
    // Determine output channel and batch index from blockIdx.z
    int oc = blockIdx.z % out_channels;
    int n = blockIdx.z / out_channels;

    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_row >= out_h || out_col >= out_w) return;

    float sum = 0.0f;
    // Loop over input channels and kernel elements
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ki = 0; ki < kernel_size; ++ki) {
            int in_row = out_row * stride - padding + ki;
            if (in_row < 0 || in_row >= in_h) continue;
            for (int kj = 0; kj < kernel_size; ++kj) {
                int in_col = out_col * stride - padding + kj;
                if (in_col < 0 || in_col >= in_w) continue;
                int input_idx = n * (in_channels * in_h * in_w) + ic * (in_h * in_w) + in_row * in_w + in_col;
                int weight_idx = oc * (in_channels * kernel_size * kernel_size) + ic * (kernel_size * kernel_size) + ki * kernel_size + kj;
                sum += input[input_idx] * __ldg(&weight[weight_idx]);
            }
        }
    }
    if (bias != nullptr) {
        sum += bias[oc];
    }
    int out_idx = n * (out_channels * out_h * out_w) + oc * (out_h * out_w) + out_row * out_w + out_col;
    output[out_idx] = sum;
}

// Host function for the forward pass
// This function dispatches the custom convolution kernel if groups == 1 and dilation == 1; otherwise it falls back to torch::conv2d.

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

    // This kernel supports only groups == 1 and dilation == 1
    if (groups != 1 || dilation != 1) {
        if (bias.has_value()) {
            return torch::conv2d(x, weight, bias.value(), {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
        } else {
            return torch::conv2d(x, weight, torch::Tensor(), {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
        }
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // assuming square kernel

    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    const int TILE_WIDTH = 16;
    const int TILE_HEIGHT = 16;
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((out_w + TILE_WIDTH - 1) / TILE_WIDTH,
              (out_h + TILE_HEIGHT - 1) / TILE_HEIGHT,
              batch_size * out_channels);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    conv2d_ldg_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, in_h, in_w,
        out_channels, kernel_size, out_h, out_w,
        stride, padding
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution using __ldg to avoid unnecessary synchronizations");
}
