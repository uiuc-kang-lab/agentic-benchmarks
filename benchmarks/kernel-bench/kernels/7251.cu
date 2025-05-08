#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define maximum constant memory size (in number of floats).
// NVIDIA GPUs usually offer 64KB constant memory; here we reserve up to 16K floats (64K bytes).
#define MAX_CONSTANT_WEIGHTS 16384

// Declare constant memory for weights.
__constant__ float const_weight[MAX_CONSTANT_WEIGHTS];

// CUDA kernel for 2D convolution using weights from constant memory.
// Assumes input tensor shape: [batch, in_channels, in_height, in_width]
// Weight tensor shape: [out_channels, in_channels / groups, k, k] (square kernel)
// Bias tensor shape: [out_channels] if provided.
__global__ void conv2d_kernel(
    const float *input,
    float *output,
    const float *bias,
    int batch,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * output_height * output_width;
    if (index >= total) return;

    // Decode the flattened index into (n, oc, oh, ow)
    int ow = index % output_width;
    int tmp = index / output_width;
    int oh = tmp % output_height;
    tmp /= output_height;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;

    // Determine group-related offsets
    int group_in_channels = in_channels / groups;
    int group_oc = out_channels / groups;  // Not used in index arithmetic but assumed to be valid.
    int group_idx = oc / group_oc;
    int input_channel_offset = group_idx * group_in_channels;

    float out_val = 0.0f;
    if (bias != nullptr) {
        out_val = bias[oc];
    }

    // Iterate over the input channels in this group and kernel elements
    for (int ic = 0; ic < group_in_channels; ic++) {
        int actual_ic = input_channel_offset + ic;
        for (int r = 0; r < kernel_size; r++) {
            int in_r = oh * stride - padding + r * dilation;
            if (in_r < 0 || in_r >= input_height) continue;
            for (int c = 0; c < kernel_size; c++) {
                int in_c = ow * stride - padding + c * dilation;
                if (in_c < 0 || in_c >= input_width) continue;
                
                int input_idx = n * (in_channels * input_height * input_width) +
                                actual_ic * (input_height * input_width) +
                                in_r * input_width + in_c;
                float input_val = input[input_idx];
                
                // Weight index: weights are stored as [oc, group_in_channels, k, k]
                int weight_idx = oc * (group_in_channels * kernel_size * kernel_size) +
                                 ic * (kernel_size * kernel_size) +
                                 r * kernel_size + c;
                float weight_val = const_weight[weight_idx];
                
                out_val += input_val * weight_val;
            }
        }
    }

    int output_idx = n * (out_channels * output_height * output_width) +
                     oc * (output_height * output_width) +
                     oh * output_width + ow;
    output[output_idx] = out_val;
}

// Host function that sets up the constant memory and launches the kernel.
// It mimics a 2D convolution with stride, padding, dilation and groups.

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    // weight tensor assumed shape: [out_channels, in_channels/groups, k, k]
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);  // square kernel
    int batch = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    // Compute output dimensions using the standard convolution formula:
    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch, out_channels, output_height, output_width}, input.options());

    // Copy the weight tensor to constant memory for faster repeated access.
    size_t weight_numel = weight.numel();
    TORCH_CHECK(weight_numel <= MAX_CONSTANT_WEIGHTS, "Weight tensor exceeds constant memory size");
    cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), weight_numel * sizeof(float));

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* bias_ptr = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);

    int total = batch * out_channels * output_height * output_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_kernel<<<blocks, threads>>>(input_ptr, output_ptr, bias_ptr,
        batch, in_channels, out_channels,
        input_height, input_width, output_height, output_width,
        kernel_size, stride, padding, dilation, groups);

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA forward function for 2D convolution using constant memory");
}
