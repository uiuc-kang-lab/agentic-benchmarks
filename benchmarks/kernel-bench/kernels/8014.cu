#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized kernel for transposed 1D convolution that uses __ldg() for read-only global memory loads.
// It assumes that the input, weight, and bias tensors are allocated with 128-bit alignment, which helps in coalescing accesses.
__global__ void aligned_ldg_conv_transposed1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int output_width,
    int groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * output_width;
    if (idx >= total) return;

    // Decode output indices: spatial position, output channel, and batch index
    int pos = idx % output_width;
    int oc = (idx / output_width) % out_channels;
    int b = idx / (output_width * out_channels);

    float sum = 0.0f;

    // Determine group parameters for grouped convolutions
    int group_in_channels = in_channels / groups;
    int group_out_channels = out_channels / groups;
    int g = oc / group_out_channels;

    // Loop over the input channels for the current group and input spatial positions
    for (int ic = 0; ic < group_in_channels; ic++) {
        int c = g * group_in_channels + ic;
        for (int j = 0; j < input_width; j++) {
            int k = pos + padding - j * stride; // compute kernel index
            if (k < 0 || k >= kernel_size) continue;
            // Use __ldg to load data from global memory (read-only) with assumed 128-bit alignment.
            float x_val = __ldg(&input[b * in_channels * input_width + c * input_width + j]);
            // Weight is stored as [in_channels, group_out_channels, kernel_size]
            int weight_idx = c * (group_out_channels * kernel_size) + (oc - g * group_out_channels) * kernel_size + k;
            float w_val = __ldg(&weight[weight_idx]);
            sum += x_val * w_val;
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += __ldg(&bias[oc]);
    }

    output[idx] = sum;
}

// Forward function to set up dimensions and launch the CUDA kernel
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
    int group_out_channels = weight.size(1);
    int out_channels = (groups == 1) ? group_out_channels : groups * group_out_channels;

    // Compute output width as per conv_transpose1d formula
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

    aligned_ldg_conv_transposed1d_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        input_width,
        kernel_size,
        stride,
        padding,
        output_width,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 1D convolution forward (CUDA) with __ldg optimization and alignment");
}
