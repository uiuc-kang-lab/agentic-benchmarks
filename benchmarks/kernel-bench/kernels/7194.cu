#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel assigns one output element per warp.
// Each warp of 32 threads cooperatively computes the convolution sum for a single output pixel.
// Each thread in the warp processes a subset of input channels and kernel elements, accumulating a partial sum in registers.
// A warp-level reduction using __shfl_down_sync aggregates these partial sums, eliminating the need for shared memory operations.

__global__ void conv2d_warp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Each warp (32 threads) computes one output element
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // Global warp index
    int lane = threadIdx.x & 31; // Thread's lane within its warp

    int total_outputs = batch * out_channels * out_height * out_width;
    if (warp_id >= total_outputs) return;

    // Map warp_id to output indices (n, oc, oh, ow)
    int tmp = warp_id;
    int n = tmp / (out_channels * out_height * out_width);
    tmp = tmp % (out_channels * out_height * out_width);
    int oc = tmp / (out_height * out_width);
    tmp = tmp % (out_height * out_width);
    int oh = tmp / out_width;
    int ow = tmp % out_width;

    float partial_sum = 0.0f;

    // Each thread in the warp loops over input channels in a strided manner
    for (int ic = lane; ic < in_channels; ic += 32) {
        // Loop over the convolution kernel spatial dimensions
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_y = oh * stride - padding + ky * dilation;
                int in_x = ow * stride - padding + kx * dilation;
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int input_idx = n * (in_channels * in_height * in_width) + ic * (in_height * in_width) + in_y * in_width + in_x;
                    int weight_idx = oc * (in_channels * kernel_size * kernel_size) + ic * (kernel_size * kernel_size) + ky * kernel_size + kx;
                    partial_sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Warp-level reduction: sum partial_sum across the 32 lanes
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    // Only lane 0 writes the final result
    if (lane == 0) {
        if (bias != nullptr) {
            partial_sum += bias[oc];
        }
        int output_idx = n * (out_channels * out_height * out_width) + oc * (out_height * out_width) + oh * out_width + ow;
        output[output_idx] = partial_sum;
    }
}

// Forward function: launches the warp-level convolution kernel if conditions allow.
// For groups != 1 or large inputs, this kernel may not be appropriate.

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

    TORCH_CHECK(groups == 1, "Only groups == 1 is supported in conv2d_warp_kernel");

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2); // assuming square kernel

    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    int total_outputs = batch * out_channels * out_height * out_width;
    // Each output element is computed by a warp (32 threads)
    int total_threads = total_outputs * 32;
    int threadsPerBlock = 256;
    int blocks = (total_threads + threadsPerBlock - 1) / threadsPerBlock;

    conv2d_warp_kernel<<<blocks, threadsPerBlock>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        in_height,
        in_width,
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
    m.def("forward", &forward, "Convolution forward function using warp-level primitives");
}
