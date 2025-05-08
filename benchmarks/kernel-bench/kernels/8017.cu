#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Modular device functions for index calculations

// Compute flattened index for input tensor [batch, in_channels, input_width]
__device__ inline int get_flat_input_index(int b, int c, int i, int in_channels, int input_width) {
    return b * in_channels * input_width + c * input_width + i;
}

// Compute flattened index for output tensor [batch, out_channels, output_width]
__device__ inline int get_flat_output_index(int b, int o, int j, int out_channels, int output_width) {
    return b * out_channels * output_width + o * output_width + j;
}

// Compute flattened index for weight tensor.
// Weight shape is assumed to be [in_channels, out_channels_per_group, kernel_size], where out_channels = out_channels_per_group * groups.
// For a given group, c ranges from c_start to c_start + group_in_channels - 1 and o ranges from o_start to o_start + group_size_out - 1.
__device__ inline int get_weight_index(int c, int o, int k, int kernel_size, int group_in_channels, int group_size_out, int c_start, int o_start) {
    // Weight index: ((c - c_start) * group_size_out + (o - o_start)) * kernel_size + k
    return ((c - c_start) * group_size_out + (o - o_start)) * kernel_size + k;
}

// Modular CUDA kernel for transposed 1D convolution
__global__ void modular_conv_transposed1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int input_width,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int groups) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * output_width;
    if (index >= total) return;

    // Decode output indices: j = spatial pos, o = output channel, b = batch index
    int j = index % output_width;
    int o = (index / output_width) % out_channels;
    int b = index / (output_width * out_channels);

    // Determine group parameters
    int group_in_channels = in_channels / groups;
    int group_size_out = out_channels / groups;
    int g = o / group_size_out;
    int c_start = g * group_in_channels;
    int o_start = g * group_size_out;

    float sum = 0.0f;
    // Loop over input channels in the group and the input spatial dimension
    for (int c = c_start; c < c_start + group_in_channels; ++c) {
        for (int i = 0; i < input_width; ++i) {
            // Compute the corresponding kernel index
            int k = j + padding - i * stride;
            if (k < 0 || k >= kernel_size) continue;
            int in_idx = get_flat_input_index(b, c, i, in_channels, input_width);
            float in_val = input[in_idx];
            int weight_idx = get_weight_index(c, o, k, kernel_size, group_in_channels, group_size_out, c_start, o_start);
            float w_val = weight[weight_idx];
            sum += in_val * w_val;
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[o];
    }

    int out_idx = get_flat_output_index(b, o, j, out_channels, output_width);
    output[out_idx] = sum;
}

// Host wrapper function
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

    int batch = x.size(0);
    int in_channels = x.size(1);
    int input_width = x.size(2);
    int kernel_size = weight.size(2);

    // For grouped convolution, weight shape is [in_channels, out_channels_per_group, kernel_size]
    int group_size_out = weight.size(1);
    int out_channels = group_size_out * groups;

    // Compute output width using formula:
    // output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto options = x.options();
    auto output = torch::zeros({batch, out_channels, output_width}, options);

    int total_threads = batch * out_channels * output_width;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    modular_conv_transposed1d_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch,
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
    m.def("forward", &forward, "Modular Transposed 1D convolution forward (CUDA) with device functions");
}
