#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define MAX_WEIGHT_SIZE 4096

__constant__ float const_weight[MAX_WEIGHT_SIZE];

__device__ inline int get_flat_input_index(int b, int c, int i, int in_channels, int input_width) {
    return b * in_channels * input_width + c * input_width + i;
}

__device__ inline int get_flat_output_index(int b, int o, int j, int out_channels, int output_width) {
    return b * out_channels * output_width + o * output_width + j;
}

__device__ inline int get_weight_index(int c, int o, int k, int kernel_size, int group_in_channels, int group_size_out, int c_start, int o_start) {
    return ((c - c_start) * group_size_out + (o - o_start)) * kernel_size + k;
}

__global__ void optimized_conv_transposed1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* bias,
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
    int o_start = g * group_size_out;

    float sum = 0.0f;
    for (int c = c_start; c < c_start + group_in_channels; ++c) {
        for (int i = 0; i < input_width; ++i) {
            int k = j + padding - i * stride;
            if (k < 0 || k >= kernel_size) continue;
            int in_idx = get_flat_input_index(b, c, i, in_channels, input_width);
            float in_val = input[in_idx];
            int weight_idx = get_weight_index(c, o, k, kernel_size, group_in_channels, group_size_out, c_start, o_start);
            float w_val = const_weight[weight_idx];
            sum += in_val * w_val;
        }
    }

    if (bias != nullptr) {
        sum += bias[o];
    }

    int out_idx = get_flat_output_index(b, o, j, out_channels, output_width);
    output[out_idx] = sum;
}

torch::Tensor forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_width = x.size(2);
    int kernel_size = weight.size(2);
    int group_size_out = weight.size(1);
    int out_channels = group_size_out * groups;

    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, output_width}, x.options());

    int num_weight_elems = weight.numel();
    TORCH_CHECK(num_weight_elems <= MAX_WEIGHT_SIZE, "Weight size exceeds constant memory limit");
    cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), num_weight_elems * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    int total_threads = batch_size * out_channels * output_width;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        auto bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_tensor.is_contiguous(), "bias must be contiguous");
        bias_ptr = bias_tensor.data_ptr<float>();
    }

    optimized_conv_transposed1d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        bias_ptr,
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
    m.def("forward", &forward, "Optimized Transposed 1D convolution forward (CUDA)");
}
