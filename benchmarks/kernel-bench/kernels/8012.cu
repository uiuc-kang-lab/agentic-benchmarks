#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Maximum size for weight in constant memory (this should be adjusted based on real constraints)
__constant__ float const_weights[4096];

// Device functions to modularize the operations

// Accumulates convolution result for a single output point
__device__ float compute_conv_point(const float* input, int b, int oc, int i, 
                                    int in_channels, int input_width, int kernel_size, 
                                    int stride, int padding, int groups, int group_size_out) {
    float sum = 0.0f;
    int group_size_in = in_channels / groups;
    int g = oc / group_size_out;  // group index

    for (int k = 0; k < kernel_size; k++) {
        int j = i + padding - k;
        if (j % stride != 0) continue;
        j /= stride;
        if (j < 0 || j >= input_width) continue;

        for (int ic = 0; ic < group_size_in; ic++) {
            int real_ic = g * group_size_in + ic;
            int input_idx = b * in_channels * input_width + real_ic * input_width + j;
            int weight_idx = (real_ic * group_size_out + (oc - g * group_size_out)) * kernel_size + k;
            sum += input[input_idx] * const_weights[weight_idx];
        }
    }
    return sum;
}

// Kernel that handles all output elements, calling device functions
__global__ void conv_transposed1d_modular_kernel(const float* input, float* output, 
                                                 const float* bias, int batch_size, 
                                                 int in_channels, int out_channels,
                                                 int input_width, int output_width,
                                                 int kernel_size, int stride,
                                                 int padding, int groups) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_width;
    if (index >= total_elements) return;

    int o = index % out_channels;
    int i = (index / out_channels) % output_width;
    int b = index / (out_channels * output_width);

    float sum = compute_conv_point(input, b, o, i, in_channels, input_width, 
                                   kernel_size, stride, padding, groups, out_channels / groups);

    if (bias != nullptr) {
        sum += bias[o];
    }

    output[index] = sum;
}

// Host function to call the kernel
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

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_width = x.size(2);
    int kernel_size = weight.size(2);
    int group_size_out = weight.size(1);
    int out_channels = group_size_out * groups;
    
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, output_width}, x.options());

    // Copy weights to constant memory
    size_t weight_size = weight.numel() * sizeof(float);
    cudaMemcpyToSymbol(const_weights, weight.data_ptr<float>(), weight_size);

    int total = batch_size * out_channels * output_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        const auto &bias_tensor = bias.value();
        CHECK_INPUT(bias_tensor);
        bias_ptr = bias_tensor.data_ptr<float>();
    }

    conv_transposed1d_modular_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Transposed 1D convolution forward (CUDA) with modular device functions");
}