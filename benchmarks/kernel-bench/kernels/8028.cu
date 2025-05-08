#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__constant__ float d_weights[1024];

// Reducing warp divergence: Unified branching
__global__ void warp_divergence_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_width,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int groups) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * output_width;
    if (index >= total) return;

    int j = index % output_width;
    int o = (index / output_width) % out_channels;
    int b = index / (output_width * out_channels);

    int group_in_channels = in_channels / groups;
    int group_size_out = out_channels / groups;
    int g = o / group_size_out;
    int c_start = g * group_in_channels;

    float sum = 0.0f;
    int valid_i_min = max(0, (j + padding + stride - kernel_size) / stride);
    int valid_i_max = min(input_width, (j + padding) / stride + 1);

    for (int i = valid_i_min; i < valid_i_max; ++i) {
        int k = j + padding - i * stride;
        for (int ic = 0; ic < group_in_channels; ++ic) {
            int input_idx = (b * in_channels + c_start + ic) * input_width + i;
            int weight_idx = ((ic * group_size_out) + (o - g * group_size_out)) * kernel_size + k;
            sum += input[input_idx] * d_weights[weight_idx];
        }
    }

    if (bias != nullptr) {
        sum += bias[o];
    }

    output[index] = sum;
}

torch::Tensor forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_width = x.size(2);
    int kernel_size = weight.size(2);
    int group_size_out = weight.size(1);
    int out_channels = group_size_out * groups;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, output_width}, x.options());

    int num_weight_elems = weight.numel();
    TORCH_CHECK(num_weight_elems <= 1024, "Weight size exceeds constant memory limit");
    cudaMemcpyToSymbol(d_weights, weight.data_ptr<float>(), num_weight_elems * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    int total_threads = batch_size * out_channels * output_width;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    warp_divergence_optimized_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_width,
        output_width,
        kernel_size,
        stride,
        padding,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 1D convolution forward (CUDA) with optimized warp divergence");
}